/* Copyright 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <atomic>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#define V2R9 209
#define CURRENT_VERSION TEST_VERSION
namespace context_test {
inline std::mutex mutex;
inline std::vector<std::string> events;
inline std::vector<std::function<void()>> pending;
inline std::atomic<int> resolutions{0};
inline std::atomic<int> lookups[4]{};
inline std::atomic<int> native_calls{0};
inline bool absent = false;
inline bool fail_resolution = false;
inline bool cache_hit = false;
inline int queue = 1;
inline std::thread::id submitter;
inline void Record(const std::string &event) {
  std::lock_guard<std::mutex> lock(mutex);
  events.push_back(event + (std::this_thread::get_id() == submitter ? ":host" : ":worker"));
}
inline void Check(bool valid, const char *message) {
  if (!valid) throw std::runtime_error(message);
}
inline int Initialize(void *ptr, bool option) {
  Check(ptr == nullptr && !option, "initialization arguments changed");
  Record("init");
  return 0;
}
inline void Release(void *ptr, bool option) {
  Check(ptr == nullptr && !option, "release arguments changed");
  Record("release");
}
inline void Uninitialize(void *ptr, bool option) {
  Check(ptr == nullptr && !option, "cleanup arguments changed");
  Record("uninit");
}
inline void CacheCleanup() { Record("cache_cleanup"); }
inline void Drain() {
  std::thread worker([] {
    for (auto &callback : pending) callback();
  });
  worker.join();
  pending.clear();
}
}  // namespace context_test

inline void *GetOpApiFuncAddr(const char *name) {
  using namespace context_test;
  if (std::string(name) == "InitHugeMemThreadLocal") {
    ++lookups[0];
    return absent ? nullptr : reinterpret_cast<void *>(Initialize);
  }
  if (std::string(name) == "ReleaseHugeMem") {
    ++lookups[1];
    return absent ? nullptr : reinterpret_cast<void *>(Release);
  }
  if (std::string(name) == "UnInitHugeMemThreadLocal") {
    ++lookups[2];
    return absent ? nullptr : reinterpret_cast<void *>(Uninitialize);
  }
  if (std::string(name) == "UnInitPTACacheThreadLocal") {
    ++lookups[3];
    return absent ? nullptr : reinterpret_cast<void *>(CacheCleanup);
  }
  throw std::runtime_error("unexpected lifecycle symbol");
}
inline void GetApiFunc(const char *, const char *, void *&execute, void *&workspace) {
  ++context_test::resolutions;
  if (context_test::fail_resolution) {
    context_test::fail_resolution = false;
    throw std::runtime_error("retry resolution");
  }
  execute = reinterpret_cast<void *>(1);
  workspace = reinterpret_cast<void *>(2);
}
inline int OpApiGetTaskQueueEnable() { return context_test::queue; }
inline void InitExecCommonCtx() {
  ++context_test::native_calls;
  auto callback = reinterpret_cast<int (*)(void *, bool)>(GetOpApiFuncAddr("InitHugeMemThreadLocal"));
  if (callback) callback(nullptr, false);
}
inline void ReleaseExecCommonCtx() {
  ++context_test::native_calls;
  auto callback = reinterpret_cast<void (*)(void *, bool)>(GetOpApiFuncAddr("ReleaseHugeMem"));
  if (callback) callback(nullptr, false);
}
inline void UnInitExecCommonCtx() {
  ++context_test::native_calls;
  auto memory = reinterpret_cast<void (*)(void *, bool)>(GetOpApiFuncAddr("UnInitHugeMemThreadLocal"));
  if (memory) memory(nullptr, false);
  auto cache = reinterpret_cast<void (*)()>(GetOpApiFuncAddr("UnInitPTACacheThreadLocal"));
  if (cache) cache();
}

// Preserve the lexical lookup, explicit lambda capture and queue ordering of PTA's macros.
#define EXEC_NPU_CMD_V1_EXT(api, ...) do {                                      \
  static void *execute = nullptr, *workspace = nullptr;                         \
  GetApiFunc(#api, #api "GetWorkspaceSize", execute, workspace);                  \
  InitExecCommonCtx();                                                         \
  if (context_test::cache_hit) break;                                          \
  auto callback = [] {                                                        \
    context_test::Record("execute");                                          \
    ReleaseExecCommonCtx();                                                   \
  };                                                                         \
  callback();                                                                \
  UnInitExecCommonCtx();                                                      \
} while (false)
#define EXEC_NPU_CMD_V2_EXT(api, ...) do {                                      \
  static void *execute = nullptr, *workspace = nullptr;                         \
  GetApiFunc(#api, #api "GetWorkspaceSize", execute, workspace);                  \
  auto callback = [] {                                                        \
    if (context_test::cache_hit) return;                                       \
    InitExecCommonCtx();                                                       \
    context_test::Record("execute");                                          \
    ReleaseExecCommonCtx();                                                   \
    UnInitExecCommonCtx();                                                    \
  };                                                                         \
  context_test::pending.push_back(callback);                                  \
} while (false)
