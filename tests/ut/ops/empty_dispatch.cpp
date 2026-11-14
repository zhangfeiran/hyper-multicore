/** Copyright 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 * Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
 */
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <random>
#include <vector>

#include "hyper_multicore/ops/runtime/empty_dispatch.h"

struct Tile {
  int32_t index;
  int32_t size;
};

int main() {
  std::mt19937 rng(6136);
  constexpr int32_t TILE = 512 * 5120;
  for (int32_t ep : {2, 16, 32, 64}) {
    for (int32_t tiles : {1, 8, 32}) {
      for (int seed = 0; seed < 24; ++seed) {
        std::vector<Tile> schedule;
        for (int32_t peer = 0; peer < ep; ++peer) {
          const int32_t boundary = static_cast<int32_t>(rng() % (tiles + 1)) * TILE;
          const int32_t size = std::clamp(boundary + (seed % 3) - 1, 0, tiles * TILE);
          for (int32_t tile = 0; tile < tiles; ++tile) schedule.push_back({tile, size});
        }
        std::shuffle(schedule.begin(), schedule.end(), rng);
        const auto active = [](const Tile &tile) { return tile.index * TILE < tile.size; };
        int32_t unprocessed = std::count_if(schedule.begin(), schedule.end(), active);
        const int32_t threshold = ep * tiles;
        int32_t credits = 0;
        for (const auto &tile : schedule) {
          if (active(tile)) {
            --unprocessed;
            ++credits;
          } else {
            const int32_t start = tile.index * TILE;
            assert(MulticoreRuntime::EmptyDispatchCredit(0, tile.index, tiles, start, tile.size, TILE) == 1);
            credits += MulticoreRuntime::EmptyDispatchCredit(
                MulticoreRuntime::BATCH_EMPTY_DISPATCH, tile.index, tiles, start, tile.size, TILE);
          }
          assert(credits <= threshold);
          assert(credits < threshold || unprocessed == 0);
        }
        assert(credits == threshold);
      }
    }
  }
}
