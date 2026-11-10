// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#define PANGU_MX_ROW_ONLY
#include "mxfp8/mx_task_graph_worker.h"
extern "C" __global__ __aicore__ void pangu_mxfp8_task_graph(GM_ADDR buffers, GM_ADDR runtime, GM_ADDR events,
                                                             GM_ADDR diagnostic, GM_ADDR launch_metadata,
                                                             GM_ADDR buffers_ref, GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  RunMxTaskGraph<MulticoreRuntime::VectorWorkerPolicy::LEGACY_ODD>(buffers, runtime, events, diagnostic,
                                                                   launch_metadata);
}
