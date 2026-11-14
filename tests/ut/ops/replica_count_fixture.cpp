/*
 * Copyright 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

#include "count_algorithm.h"

template <typename InputT>
int CheckRows() {
  std::mt19937 generator(7391);
  int checked = 0;
  for (int experts : {1, 2, 8, 192, 256, 384, 512}) {
    std::uniform_int_distribution<int> random_id(-3, experts + 3);
    for (int trial = 0; trial < 1000; ++trial) {
      std::array<InputT, 8> ids;
      for (int index = 0; index < 8; ++index) {
        ids[index] = trial % 2 ? random_id(generator) : (trial * 13 + index * 11) % experts;
      }
      if (trial == 0) ids.fill(std::numeric_limits<InputT>::min());
      if (trial == 1) ids.fill(std::numeric_limits<InputT>::max());
      std::array<int64_t, 513> generic = {}, fixed = {};
      hot::CountRow(ids.data(), generic.data(), 8, experts);
      hot::CountFixedRow<8>(ids.data(), fixed.data(), experts);
      if (generic != fixed) throw std::runtime_error("fixed TopK changed counts or validation flag");
      ++checked;
    }
  }
  return checked;
}

int main() {
  const int checked = CheckRows<int32_t>() + CheckRows<int64_t>();
  std::cout << "Verified " << checked << " exact fixed-TopK histograms and invalid-ID flags\n";
}
