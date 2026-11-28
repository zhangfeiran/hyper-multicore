// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace {
bool StorageIsUnique(const at::Tensor& owner) {
  return owner.storage().use_count() == 1;
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(hyper_multicore, m) {
  m.def("_storage_is_unique(Tensor owner) -> bool", &StorageIsUnique);
}
