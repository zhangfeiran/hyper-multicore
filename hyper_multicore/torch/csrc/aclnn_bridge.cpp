// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "aclnn_bridge.h"

#include <ATen/ops/empty.h>
#include <dlfcn.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include <limits>
#include <mutex>
#include <string>

namespace hyper_multicore {
namespace {

struct VendorLibrary {
  std::mutex mutex;
  std::string path;
  void *handle = nullptr;
};

VendorLibrary &Vendor() {
  static VendorLibrary library;
  return library;
}

void *OpenLibrary(const char *path) {
  auto *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  TORCH_CHECK(handle != nullptr, "cannot open ACLNN library ", path, ": ", dlerror());
  return handle;
}

void ConfigureVendor(std::string path) {
  auto &vendor = Vendor();
  std::lock_guard<std::mutex> lock(vendor.mutex);
  if (vendor.handle != nullptr) {
    TORCH_CHECK(path == vendor.path, "the ACLNN vendor is already bound to ", vendor.path);
    return;
  }
  vendor.handle = OpenLibrary(path.c_str());
  vendor.path = std::move(path);
}

aclDataType AclnnType(at::ScalarType type) {
  switch (type) {
    case at::kBool: return ACL_BOOL;
    case at::kByte: return ACL_UINT8;
    case at::kChar: return ACL_INT8;
    case at::kShort: return ACL_INT16;
    case at::kInt: return ACL_INT32;
    case at::kLong: return ACL_INT64;
    case at::kHalf: return ACL_FLOAT16;
    case at::kBFloat16: return ACL_BF16;
    case at::kFloat: return ACL_FLOAT;
    case at::kDouble: return ACL_DOUBLE;
    case at::kUInt16: return ACL_UINT16;
    case at::kUInt32: return ACL_UINT32;
    case at::kUInt64: return ACL_UINT64;
    case at::kFloat8_e4m3fn: return ACL_FLOAT8_E4M3FN;
    case at::kFloat8_e5m2: return ACL_FLOAT8_E5M2;
    default: TORCH_CHECK(false, "unsupported ACLNN tensor dtype ", type);
  }
}

aclFormat AclnnFormat(const at::Tensor &tensor) {
  const auto format = at_npu::native::get_npu_format(tensor);
  TORCH_CHECK(format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW || format == ACL_FORMAT_NHWC ||
                  format == ACL_FORMAT_NCL || format == ACL_FORMAT_NCDHW || format == ACL_FORMAT_NDHWC,
              "the public ACLNN bridge requires a base-format tensor, got ", format);
  switch (tensor.dim()) {
    case 3: return ACL_FORMAT_NCL;
    case 4: return ACL_FORMAT_NCHW;
    case 5: return ACL_FORMAT_NCDHW;
    default: return ACL_FORMAT_ND;
  }
}

}  // namespace

AclnnEntry ResolveAclnn(const char *execute, const char *prepare) {
  auto &vendor = Vendor();
  void *handle;
  {
    std::lock_guard<std::mutex> lock(vendor.mutex);
    TORCH_CHECK(vendor.handle != nullptr, "the HyperMulticore ACLNN vendor is not configured");
    handle = vendor.handle;
  }
  auto *launch = dlsym(handle, execute);
  auto *workspace = dlsym(handle, prepare);
  if (launch == nullptr && workspace == nullptr) {
    static auto *standard = OpenLibrary("libopapi.so");
    launch = dlsym(standard, execute);
    workspace = dlsym(standard, prepare);
  }
  TORCH_CHECK(launch != nullptr && workspace != nullptr, "cannot resolve ACLNN entry points for ", execute);
  return {workspace, reinterpret_cast<AclnnExecute>(launch)};
}

c10::Device AclnnDevice(const at::Tensor &first) {
  TORCH_CHECK(first.defined() && first.device().is_privateuseone(), "ACLNN requires an NPU tensor");
  return first.device();
}

c10::Device AclnnDevice(at::TensorList first) {
  TORCH_CHECK(!first.empty(), "ACLNN requires a nonempty tensor list as its first operand");
  return AclnnDevice(first.front());
}

AclnnArguments::TensorDescriptor AclnnArguments::MakeTensor(const at::Tensor &tensor) {
  if (!tensor.defined()) {
    return {};
  }
  TORCH_CHECK(tensor.device() == stream.device(), "all ACLNN operands must be on the launch device");
  const int64_t storage_elements = tensor.storage().nbytes() / tensor.element_size();
  TensorDescriptor native(aclCreateTensor(
      tensor.sizes().data(), tensor.dim(), AclnnType(tensor.scalar_type()), tensor.strides().data(),
      tensor.storage_offset(), AclnnFormat(tensor), &storage_elements, 1,
      const_cast<void *>(tensor.storage().data())));
  TORCH_CHECK(native != nullptr, "aclCreateTensor failed: ", aclGetRecentErrMsg());
  storage_.push_back(tensor);
  c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
  return native;
}

aclTensor *AclnnArguments::Convert(const at::Tensor &tensor) {
  tensors_.push_back(MakeTensor(tensor));
  return tensors_.back().get();
}

aclTensor *AclnnArguments::Convert(const c10::optional<at::Tensor> &tensor) {
  return tensor.has_value() ? Convert(*tensor) : nullptr;
}

aclTensorList *AclnnArguments::Convert(at::TensorList tensors) {
  std::vector<TensorDescriptor> descriptors;
  std::vector<const aclTensor *> values;
  descriptors.reserve(tensors.size());
  values.reserve(tensors.size());
  for (const auto &tensor : tensors) {
    descriptors.push_back(MakeTensor(tensor));
    values.push_back(descriptors.back().get());
  }
  AclObject<aclTensorList, aclDestroyTensorList> native(aclCreateTensorList(values.data(), values.size()));
  TORCH_CHECK(native != nullptr, "aclCreateTensorList failed: ", aclGetRecentErrMsg());
  for (auto &descriptor : descriptors) {
    descriptor.release();
  }
  lists_.push_back(std::move(native));
  return lists_.back().get();
}

aclIntArray *AclnnArguments::Convert(const c10::optional<at::IntArrayRef> &value) {
  if (!value.has_value()) {
    return nullptr;
  }
  AclObject<aclIntArray, aclDestroyIntArray> native(aclCreateIntArray(value->data(), value->size()));
  TORCH_CHECK(native != nullptr, "aclCreateIntArray failed: ", aclGetRecentErrMsg());
  arrays_.push_back(std::move(native));
  return arrays_.back().get();
}

void SubmitAclnn(const char *name, AclnnExecute execute, uint64_t bytes,
                 std::shared_ptr<AclnnArguments> arguments) {
  TORCH_CHECK(bytes <= std::numeric_limits<int64_t>::max(), "ACLNN workspace size exceeds int64");
  at::Tensor workspace;
  if (bytes != 0) {
    workspace = at::empty({static_cast<int64_t>(bytes)}, at::device(arguments->stream.device()).dtype(at::kByte));
  }
  const auto stream = arguments->stream.stream(false);
  auto run = [arguments = std::move(arguments), workspace, execute, bytes, stream]() -> int {
    return execute(workspace.defined() ? workspace.data_ptr() : nullptr, bytes, arguments->executor.release(), stream);
  };
  at_npu::native::OpCommand::RunOpApi(name, std::move(run));
}

TORCH_LIBRARY_FRAGMENT(hyper_multicore, m) {
  m.def("_configure_aclnn_vendor(str path) -> ()", &ConfigureVendor);
}

}  // namespace hyper_multicore
