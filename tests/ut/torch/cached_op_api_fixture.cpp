/* Copyright 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cached_op_api.h"
#include <iostream>

void LaunchFirst() { HYPER_EXEC_NPU_CMD_EXT(FirstOp, 1); }
void LaunchSecond() { HYPER_EXEC_NPU_CMD_EXT(SecondOp, 2); }

std::vector<std::string> ExpectedEvents() {
  using namespace context_test;
  std::vector<std::string> expected;
  if (cache_hit) {
    if (!absent && queue == 1) expected = {"init:host", "init:host"};
    return expected;
  }
  const auto suffix = queue == 1 ? ":host" : ":worker";
  for (int op = 0; op < 2; ++op) {
    if (!absent) expected.push_back(std::string("init") + suffix);
    expected.push_back(std::string("execute") + suffix);
    if (!absent) {
      for (const auto *name : {"release", "uninit", "cache_cleanup"}) {
        expected.push_back(std::string(name) + suffix);
      }
    }
  }
  return expected;
}

int main(int argc, char **argv) {
  using namespace context_test;
  try {
    Check(argc == 2, "expected callback availability mode");
    absent = std::string(argv[1]) == "absent";
    submitter = std::this_thread::get_id();
    fail_resolution = true;
    try {
      LaunchFirst();
      throw std::runtime_error("missing symbol-resolution failure");
    } catch (const std::runtime_error &error) {
      Check(std::string(error.what()) == "retry resolution", "wrong initial failure");
    }
    Check(events.empty(), "failed resolution invoked a lifecycle callback");
    for (queue = 1; queue <= 2; ++queue) {
      for (int iteration = 0; iteration < 3; ++iteration) {
        cache_hit = iteration == 1;
        events.clear();
        LaunchFirst();
        LaunchSecond();
        if (queue == 2) {
          Check(events.empty(), "V2 ran context callbacks on the submitting thread");
          Drain();
        }
        const auto expected = ExpectedEvents();
        Check(events == expected, "context order, cache-hit behavior, or callback thread changed");
      }
    }
    Check(resolutions == 5, "operator symbols were not resolved once per queue and call site after retry");
#if TEST_VERSION == V2R9
    Check(native_calls == 0, "audited context path still enters uncached framework helpers");
    for (const auto &count : lookups) Check(count == 1, "lifecycle symbol was resolved repeatedly");
#else
    Check(native_calls > 0 && lookups[0] > 1, "unaudited PTA version bypassed native context helpers");
#endif
    std::cout << "Verified V1/V2 lifecycle order, async thread, cache hits, null callbacks and resolution retry\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
