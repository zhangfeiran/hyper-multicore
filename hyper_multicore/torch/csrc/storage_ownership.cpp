// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace {
bool StorageIsUnique(at::TensorList owners) {
  for (const auto& owner : owners) {
    if (owner.storage().use_count() != 1) {
      return false;
    }
  }
  return true;
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(hyper_multicore, m) {
  m.def("_storage_is_unique(Tensor[] owners) -> bool", &StorageIsUnique);
}
