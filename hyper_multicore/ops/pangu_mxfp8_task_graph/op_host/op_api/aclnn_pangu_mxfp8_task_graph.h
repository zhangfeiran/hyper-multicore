// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef ACLNN_PANGU_MXFP8_TASK_GRAPH_H_
#define ACLNN_PANGU_MXFP8_TASK_GRAPH_H_
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
extern "C" {
ACLNN_API aclnnStatus aclnnPanguMxfp8TaskGraphGetWorkspaceSize(const aclTensorList *buffers,
                                                               const aclTensor *runtime_config, const aclTensor *events,
                                                               const aclTensor *diagnostic,
                                                               const aclTensor *launch_metadata,
                                                               const aclTensorList *buffers_ref, int64_t buffer_count,
                                                               uint64_t *workspaceSize, aclOpExecutor **executor);
ACLNN_API aclnnStatus aclnnPanguMxfp8TaskGraph(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                               aclrtStream stream);
}
#endif
