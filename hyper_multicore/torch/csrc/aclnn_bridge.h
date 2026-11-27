// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef HYPER_MULTICORE_TORCH_CSRC_ACLNN_BRIDGE_H_
#define HYPER_MULTICORE_TORCH_CSRC_ACLNN_BRIDGE_H_

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_mhc_pre_sinkhorn.h>
#include <aclnnop/aclnn_moe_token_permute.h>
#include <aclnnop/aclnn_moe_token_permute_grad.h>
#include <aclnnop/aclnn_moe_token_unpermute_grad.h>

#include "aclnn_hyper_mega_gate_route.h"
#include "aclnn_hyper_mega_gate_route_grad.h"
#include "aclnn_hyper_mega_mhc.h"
#include "aclnn_hyper_mega_mhc_grad.h"
#include "aclnn_hyper_mega_moe.h"
#include "aclnn_hyper_mega_moe_grad.h"
#include "aclnn_hyper_mega_moe_mxfp8.h"
#include "aclnn_hyper_mega_moe_mxfp8_grad.h"

#include <ATen/core/Tensor.h>
#include <c10/core/DeviceGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <memory>
#include <tuple>
#include <type_traits>
#include <vector>

namespace hyper_multicore {

using AclnnExecute = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

struct AclnnEntry {
  void *prepare;
  AclnnExecute execute;
};

AclnnEntry ResolveAclnn(const char *execute, const char *prepare);

template <typename T, auto Destroy>
struct AclDeleter {
  void operator()(T *value) const noexcept { Destroy(value); }
};

template <typename T, auto Destroy>
using AclObject = std::unique_ptr<T, AclDeleter<T, Destroy>>;

// Descriptors, tensor storage and the executor survive until PTA submits the
// command. CANN consumes a non-repeatable executor on the second API call.
class AclnnArguments {
 public:
  explicit AclnnArguments(c10_npu::NPUStream stream) : stream(stream) {}
  ~AclnnArguments() { executor.reset(); }  // Release an unsubmitted executor before its descriptors.

  aclTensor *Convert(const at::Tensor &tensor);
  aclTensor *Convert(const c10::optional<at::Tensor> &tensor);
  aclTensorList *Convert(at::TensorList tensors);
  aclIntArray *Convert(const c10::optional<at::IntArrayRef> &value);

  template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
  T Convert(T value) { return value; }

  c10_npu::NPUStream stream;
  AclObject<aclOpExecutor, aclDestroyAclOpExecutor> executor;

 private:
  using TensorDescriptor = AclObject<aclTensor, aclDestroyTensor>;

  TensorDescriptor MakeTensor(const at::Tensor &tensor);

  std::vector<at::Tensor> storage_;
  std::vector<TensorDescriptor> tensors_;
  std::vector<AclObject<aclTensorList, aclDestroyTensorList>> lists_;
  std::vector<AclObject<aclIntArray, aclDestroyIntArray>> arrays_;
};

c10::Device AclnnDevice(const at::Tensor &first);
c10::Device AclnnDevice(at::TensorList first);
void SubmitAclnn(const char *name, AclnnExecute execute, uint64_t bytes,
                 std::shared_ptr<AclnnArguments> arguments);

template <typename Prepare, typename First, typename... Args>
void EnqueueAclnn(const char *name, const AclnnEntry &entry, Prepare prepare,
                  const First &first, const Args &...args) {
  const auto device = AclnnDevice(first);
  c10::DeviceGuard guard(device);
  auto arguments = std::make_shared<AclnnArguments>(c10_npu::getCurrentNPUStream(device.index()));
  auto parameters = std::tuple{arguments->Convert(first), arguments->Convert(args)...};
  uint64_t bytes = 0;
  aclOpExecutor *executor = nullptr;
  const auto status = std::apply([&](auto... items) {
    return prepare(items..., &bytes, &executor);
  }, parameters);
  arguments->executor.reset(executor);
  TORCH_CHECK(status == ACL_SUCCESS, name, "GetWorkspaceSize failed: ", status, " ", aclGetRecentErrMsg());
  SubmitAclnn(name, entry.execute, bytes, std::move(arguments));
}

}  // namespace hyper_multicore

#endif  // HYPER_MULTICORE_TORCH_CSRC_ACLNN_BRIDGE_H_
