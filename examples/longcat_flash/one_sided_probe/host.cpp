/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>
#include <string>

#include "acl/acl.h"
#include "shmem.h"
#include "host_device/shmem_common_types.h"

extern aclshmem_device_host_state_t g_state;
extern void launch_one_sided_probe(void *, void *, void *, void *, uint32_t, int32_t, int32_t, int32_t, uint64_t);

namespace {
struct Context {
  aclrtStream stream{};
  aclrtEvent begin{};
  aclrtEvent end{};
  void *source{};
  void *destination{};
  int32_t *control{};
  uint64_t capacity{};
  uint64_t pipeline_bytes{};
  int32_t rank{};
  int32_t size{};
  bool initialized{false};
} context;
std::string last_error;

int checked(int status, const char *operation) {
  if (status != 0) {
    last_error = std::string(operation) + " failed: " + std::to_string(status);
  }
  return status;
}
}  // namespace

extern "C" const char *probe_last_error() { return last_error.c_str(); }
extern "C" int probe_initialized() { return g_state.is_aclshmem_initialized ? 1 : 0; }
extern "C" uint64_t probe_unique_id_size() { return sizeof(aclshmemx_uniqueid_t); }
extern "C" uint32_t probe_transport_bit(int roce) {
  return roce ? ACLSHMEM_TRANSPORT_ROCE : ACLSHMEM_TRANSPORT_MTE;
}
extern "C" uint32_t probe_peer_transport(int peer) {
  return peer >= 0 && peer < g_state.npes ? g_state.topo_list[peer] : 0;
}

extern "C" int probe_unique_id(void *output) {
  int status = checked(aclshmemx_set_conf_store_tls(false, nullptr, 0), "disable_store_tls");
  return status ? status : checked(aclshmemx_get_uniqueid(static_cast<aclshmemx_uniqueid_t *>(output)), "unique_id");
}

extern "C" int probe_open(int rank, int size, const void *id, uint64_t heap_bytes,
                           uint64_t capacity, int roce) {
  aclshmemx_init_attr_t attr{};
  aclshmemx_uniqueid_t uid{};
  std::memcpy(&uid, id, sizeof(uid));
  attr.option_attr.version = static_cast<int32_t>((1U << 16U) + sizeof(aclshmem_init_optional_attr_t));
  attr.option_attr.data_op_engine_type = roce ? ACLSHMEM_DATA_OP_ROCE : ACLSHMEM_DATA_OP_MTE;
  attr.option_attr.sockFd = -1;
  attr.option_attr.shm_init_timeout = 60;
  attr.option_attr.shm_create_timeout = 60;
  attr.option_attr.control_operation_timeout = 60;
  int status = checked(aclshmemx_set_conf_store_tls(false, nullptr, 0), "disable_store_tls");
  if (status) return status;
  status = checked(aclshmemx_set_attr_uniqueid_args(rank, size, heap_bytes, &uid, &attr), "unique_id_args");
  if (status) return status;
  attr.option_attr.data_op_engine_type = roce ? ACLSHMEM_DATA_OP_ROCE : ACLSHMEM_DATA_OP_MTE;
  status = checked(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, &attr), "shmem_init");
  if (status) return status;
  context.initialized = true;
  context.capacity = capacity;
  context.rank = rank;
  context.size = size;
  context.source = aclshmem_align(512, capacity);
  context.destination = aclshmem_align(512, capacity);
  context.control = static_cast<int32_t *>(aclshmem_align(512, 4096));
  if (!context.source || !context.destination || !context.control) {
    last_error = "symmetric allocation failed";
    return -1;
  }
  status = checked(aclrtMemset(context.control, 4096, 0, 4096), "control_init");
  if (status) return status;
  status = checked(aclrtCreateStream(&context.stream), "create_stream");
  if (status) return status;
  status = checked(aclrtCreateEvent(&context.begin), "create_begin_event");
  return status ? status : checked(aclrtCreateEvent(&context.end), "create_end_event");
}

extern "C" int probe_prepare(const void *source, uint64_t bytes, int32_t generation) {
  if (bytes > context.capacity) return checked(-1, "payload_capacity");
  int status = 0;
  if (bytes) status = checked(aclrtMemcpy(context.source, context.capacity, source, bytes,
                                         ACL_MEMCPY_HOST_TO_DEVICE), "source_upload");
  if (status) return status;
  status = checked(aclrtMemcpy(context.control + 32, sizeof(generation), &generation, sizeof(generation),
                              ACL_MEMCPY_HOST_TO_DEVICE), "tag_upload");
  if (status) return status;
  return checked(aclrtMemset(context.destination, context.capacity, 0xA5,
                             bytes ? bytes : 32), "destination_init");
}

extern "C" int probe_step(uint32_t bytes, int32_t peer, int32_t generation, int32_t mode,
                           uint64_t wait_cycles, float *milliseconds, uint64_t *results) {
  int status = checked(aclrtRecordEvent(context.begin, context.stream), "record_begin");
  if (status) return status;
  static_cast<void>(aclrtGetLastError(ACL_RT_THREAD_LEVEL));
  launch_one_sided_probe(context.stream, context.destination, context.source, context.control,
                         bytes, peer, generation, mode, wait_cycles);
  status = checked(aclrtGetLastError(ACL_RT_THREAD_LEVEL), "launch");
  if (status) return status;
  status = checked(aclrtRecordEvent(context.end, context.stream), "record_end");
  if (status) return status;
  status = checked(aclrtSynchronizeStream(context.stream), "synchronize");
  if (status) return status;
  status = checked(aclrtEventElapsedTime(milliseconds, context.begin, context.end), "elapsed");
  if (status) return status;
  return checked(aclrtMemcpy(results, 160, context.control + 128, 160, ACL_MEMCPY_DEVICE_TO_HOST), "results");
}

extern "C" int probe_read(void *destination, uint64_t bytes) {
  return checked(aclrtMemcpy(destination, bytes, context.destination, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "readback");
}

extern "C" int probe_close() {
  if (context.control) aclshmem_free(context.control);
  if (context.destination) aclshmem_free(context.destination);
  if (context.source) aclshmem_free(context.source);
  if (context.begin) aclrtDestroyEvent(context.begin);
  if (context.end) aclrtDestroyEvent(context.end);
  if (context.stream) aclrtDestroyStream(context.stream);
  int status = context.initialized ? checked(aclshmem_finalize(), "shmem_finalize") : 0;
  context = {};
  return status;
}

extern void launch_pipeline_probe(void *, uint8_t *, uint8_t *, uint8_t *, uint32_t, int32_t, int32_t, uint64_t);

extern "C" int probe_pipeline_configure() {
  if (context.size != 16 || context.capacity != 2048ULL * 6144 * 2) {
    return checked(-1, "pipeline requires EP16 and the original LongCat source shape");
  }
  context.pipeline_bytes = 8ULL * 8 * 128 * 6144 * 2;
  aclshmem_free(context.destination);
  context.destination = aclshmem_align(512, context.pipeline_bytes);
  return context.destination ? 0 : checked(-1, "pipeline receive allocation");
}

extern "C" int probe_pipeline_prepare(const void *source, int32_t generation) {
  if (generation <= 0 || generation >= (1 << 23)) return checked(-1, "credit generation");
  int status = probe_prepare(source, context.capacity, generation);
  if (status) return status;
  status = checked(aclrtMemset(context.destination, context.pipeline_bytes, 0xA5,
                              context.pipeline_bytes), "pipeline receive init");
  if (status) return status;
  status = checked(aclrtMemset(context.control, 1024, 0, 1024), "pipeline credit init");
  if (status) return status;
  int32_t tags[8];
  for (int expert = 0; expert < 8; ++expert) {
    tags[expert] = (generation << 8) | ((1 << (expert + 1)) - 1);
  }
  return checked(aclrtMemcpy(context.control + 256, sizeof(tags), tags, sizeof(tags),
                             ACL_MEMCPY_HOST_TO_DEVICE), "immutable credit records");
}

extern "C" int probe_pipeline_step(uint32_t rows, int32_t generation, uint64_t wait_cycles,
                                    float *milliseconds, uint64_t *results) {
  if (rows > 128) return checked(-1, "pipeline row limit");
  int status = checked(aclrtRecordEvent(context.begin, context.stream), "pipeline begin");
  if (status) return status;
  static_cast<void>(aclrtGetLastError(ACL_RT_THREAD_LEVEL));
  launch_pipeline_probe(context.stream, static_cast<uint8_t *>(context.destination),
                        static_cast<uint8_t *>(context.source), reinterpret_cast<uint8_t *>(context.control),
                        rows, context.rank, generation, wait_cycles);
  status = checked(aclrtGetLastError(ACL_RT_THREAD_LEVEL), "pipeline launch");
  if (status) return status;
  status = checked(aclrtRecordEvent(context.end, context.stream), "pipeline end");
  if (status) return status;
  status = checked(aclrtSynchronizeStream(context.stream), "pipeline synchronize");
  if (status) return status;
  status = checked(aclrtEventElapsedTime(milliseconds, context.begin, context.end), "pipeline elapsed");
  if (status) return status;
  return checked(aclrtMemcpy(results, 4096, context.control, 4096,
                             ACL_MEMCPY_DEVICE_TO_HOST), "pipeline observations");
}
