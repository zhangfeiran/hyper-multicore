/** Copyright 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 * Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
 */
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include "hyper_multicore/ops/runtime/gradient_order_policy.h"
int main(int argc, char **argv) {
  const uint32_t policy = static_cast<uint32_t>(std::stoul(argv[1]));
  std::vector<int64_t> rows;
  for (int i = 2; i < argc; ++i) rows.push_back(std::stoll(argv[i]));
  uint32_t order[MulticoreRuntime::GRADIENT_ORDER_MAX_EXPERTS] = {};
  bool join = MulticoreRuntime::BuildGradientExpertOrder(rows.data(), rows.size(), policy, order);
  std::cout << join;
  for (uint32_t i = 0; i < rows.size(); ++i) std::cout << ' ' << order[i];
  std::cout << '\n';
}
