// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef HYPER_MULTICORE_OPS_RUNTIME_CORE_TYPE_H_
#define HYPER_MULTICORE_OPS_RUNTIME_CORE_TYPE_H_

#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510 && defined(__DAV_CUBE__)) || \
    ((!defined(__NPU_ARCH__) || __NPU_ARCH__ != 3510) && defined(__DAV_C220_CUBE__))
#define HMC_RUNTIME_CUBE
#endif

#endif  // HYPER_MULTICORE_OPS_RUNTIME_CORE_TYPE_H_
