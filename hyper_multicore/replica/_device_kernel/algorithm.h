// Copyright 2026 Huawei Technologies Co., Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================
#pragma once
#include <stdint.h>
#ifndef HOT_DEVICE
#define HOT_DEVICE inline
#define HOT_LOCAL
#define HOT_OUTPUT
#endif
namespace hot {
HOT_DEVICE int64_t Min(int64_t a, int64_t b) { return a < b ? a : b; }
HOT_DEVICE int64_t Max(int64_t a, int64_t b) { return a > b ? a : b; }
class Solver {
 public:
  HOT_DEVICE Solver(HOT_LOCAL int64_t *memory, int ranks, int experts, int budget)
      : r(ranks),
        e(experts),
        h(experts / ranks),
        b(budget),
        n(ranks * experts),
        w(h + b),
        counts(memory),
        copies(counts + n),
        trial(copies + n),
        candidates(trial + n),
        edges(candidates + n),
        totals(edges + n),
        remaining(totals + e),
        segments(remaining + e),
        loads(segments + e),
        original(loads + r),
        targets(original + r),
        headroom(targets + r) {}

  HOT_DEVICE int Run(HOT_OUTPUT int64_t *control, HOT_OUTPUT int32_t *dispatch, int64_t capacity, int64_t target,
                     int64_t minimum, int64_t admission = 0, int64_t saved_rows = 0) {
    int64_t sum = 0;
    // Bound products as well as sums; all ordinary TopK workloads fit far below this limit.
    const int64_t safe_count = INT64_MAX / (n * Max(e, r) * 4LL);
    for (int i = 0; i < n; ++i) {
      if (counts[i] < 0 || counts[i] > safe_count) {
        control[0] = 1;
        return 1;
      }
      sum += counts[i];
      copies[i] = 0;
    }
    for (int j = 0; j < e; ++j) {
      totals[j] = 0;
      for (int k = 0; k < r; ++k) totals[j] += counts[k * e + j];
      copies[(j / h) * e + j] = totals[j];
      remaining[j] = totals[j];
    }
    Refresh(copies);
    int64_t peak = 0;
    for (int k = 0; k < r; ++k) {
      original[k] = loads[k];
      peak = Max(peak, loads[k]);
    }
    int64_t average = (sum + r - 1) / r;
    target = Max(target, average);
    int64_t limit = capacity < 0 ? PruningCapacity(peak, average) : capacity;
    if (b && peak > target && AdmitsPeak(average, admission, saved_rows, peak, limit)) {
      Balance(sum, target, minimum, peak, limit);
    }
    Admit(admission, saved_rows, peak, limit);
    return Materialize(control, dispatch, capacity);
  }

 private:
  HOT_DEVICE void Balance(int64_t sum, int64_t target, int64_t minimum, int64_t peak, int64_t limit) {
    Construct(sum);
    Refresh(copies);
    int64_t current_peak = 0;
    for (int k = 0; k < r; ++k) current_peak = Max(current_peak, loads[k]);
    if (current_peak > peak) {
      for (int k = 0; k < r; ++k)
        for (int j = 0; j < e; ++j) copies[k * e + j] = j / h == k ? totals[j] : 0;
      Refresh(copies);
    }
    Improve(target);
    if (minimum) {
      Prune(minimum, Min(peak, limit));
      Rebalance();
    }
  }

  HOT_DEVICE bool AdmitsPeak(int64_t candidate, int64_t admission, int64_t saved_rows, int64_t peak, int64_t limit) {
    if ((!admission && !saved_rows) || peak > limit) return true;
    // Quotient/remainder arithmetic avoids multiplying a large receive count by basis points.
    const int64_t required = (peak / 10000) * admission + ((peak % 10000) * admission + 9999) / 10000;
    return peak - candidate >= Max(required, saved_rows);
  }

  HOT_DEVICE void Admit(int64_t admission, int64_t saved_rows, int64_t peak, int64_t limit) {
    int64_t current_peak = 0;
    for (int k = 0; k < r; ++k) current_peak = Max(current_peak, loads[k]);
    if (AdmitsPeak(current_peak, admission, saved_rows, peak, limit)) return;
    for (int k = 0; k < r; ++k)
      for (int j = 0; j < e; ++j) copies[k * e + j] = j / h == k ? totals[j] : 0;
    Refresh(copies);
  }

  HOT_DEVICE void Refresh(HOT_LOCAL int64_t *data) {
    for (int k = 0; k < r; ++k) {
      loads[k] = 0;
      for (int j = 0; j < e; ++j) loads[k] += data[k * e + j];
    }
  }
  HOT_DEVICE int64_t PruningCapacity(int64_t peak, int64_t average) {
    int64_t source_total = 0, largest = 0;
    for (int j = 0; j < e; ++j) source_total += counts[j];
    bool equal = true;
    for (int k = 0; k < r; ++k) {
      int64_t local = 0;
      for (int j = 0; j < e; ++j) {
        local += counts[k * e + j];
        largest = Max(largest, counts[k * e + j]);
      }
      equal = equal && local == source_total;
    }
    int budget = Min(b, h);
    if (largest && equal) {
      int64_t top_k = Min(e, source_total / largest);
      while (source_total % top_k) --top_k;
      int64_t upper = r * (source_total / top_k) * Min(top_k, h);
      if (budget == h) return source_total;
      if (budget) upper = Min(upper, ((h - budget) * upper + budget * source_total + h - 1) / h + r - 1);
      return upper;
    }
    if (budget == h) return average;
    return Min(peak, ((h - budget) * peak + budget * average + h - 1) / h + r - 1);
  }
  HOT_DEVICE void Construct(int64_t sum) {
    for (int k = 0; k < r; ++k) targets[k] = sum / r + (k < sum % r);
    for (int round = 0; round < r - 1; ++round) {
      int donor = 0, receiver = 0;
      for (int k = 1; k < r; ++k) {
        if (loads[k] - targets[k] > loads[donor] - targets[donor]) donor = k;
        if (loads[k] - targets[k] < loads[receiver] - targets[receiver]) receiver = k;
      }
      int64_t amount = targets[receiver] - loads[receiver];
      if (!amount) break;
      for (int j = 0; j < e; ++j) segments[j] = 0;
      int64_t pending = amount;
      for (int i = 0; i < h && pending; ++i) {
        int best = donor * h;
        for (int j = best + 1; j < (donor + 1) * h; ++j)
          if (remaining[j] > remaining[best]) best = j;
        int64_t moved = Min(pending, remaining[best]);
        segments[best] += moved;
        remaining[best] -= moved;
        pending -= moved;
      }
      int64_t quota = Min(b, h) * amount / h;
      for (int i = 0; i < Min(b, h) && quota; ++i) {
        int best = donor * h;
        for (int j = best + 1; j < (donor + 1) * h; ++j)
          if (segments[j] > segments[best]) best = j;
        int64_t moved = Min(segments[best], quota);
        copies[donor * e + best] -= moved;
        copies[receiver * e + best] += moved;
        segments[best] = 0;
        quota -= moved;
      }
      loads[donor] -= amount;
      loads[receiver] += amount;
    }
  }
  HOT_DEVICE void Improve(int64_t target) {
    while (true) {
      int donor = -1, receiver = -1, expert = -1;
      int64_t moved = 0;
      // Choose the highest-load donor that has a legal move, then that donor's best edge.
      for (int d = 0; d < r; ++d) {
        if (loads[d] <= target || (donor >= 0 && loads[d] <= loads[donor])) continue;
        int local_receiver = -1, local_expert = -1;
        int64_t local_moved = 0;
        bool local_existing = false;
        for (int t = 0; t < r; ++t) {
          if (loads[t] >= target) continue;
          int guests = 0;
          for (int j = 0; j < e; ++j)
            if (j / h != t && copies[t * e + j] > 0) ++guests;
          for (int j = d * h; j < (d + 1) * h; ++j) {
            bool existing = copies[t * e + j] > 0;
            if (!existing && guests >= b) continue;
            int64_t rows = Min(copies[d * e + j], Min(loads[d] - target, target - loads[t]));
            if (rows > local_moved || (rows && rows == local_moved && existing && !local_existing)) {
              local_receiver = t;
              local_expert = j;
              local_moved = rows;
              local_existing = existing;
            }
          }
        }
        if (local_moved) {
          donor = d;
          receiver = local_receiver;
          expert = local_expert;
          moved = local_moved;
        }
      }
      if (!moved) return;
      copies[donor * e + expert] -= moved;
      copies[receiver * e + expert] += moved;
      loads[donor] -= moved;
      loads[receiver] += moved;
    }
  }
  HOT_DEVICE int SortedGuests(HOT_LOCAL int64_t *order, int owner, int excluded, bool ascending, int64_t threshold) {
    int length = 0;
    for (int k = 0; k < r; ++k)
      for (int j = 0; j < e; ++j) {
        int index = k * e + j;
        int64_t rows = copies[index];
        if (j / h == k || !rows || rows >= threshold || index == excluded || (owner >= 0 && j / h != owner)) continue;
        int at = length++;
        while (at > 0 && (ascending ? rows < copies[order[at - 1]] : rows > copies[order[at - 1]])) {
          order[at] = order[at - 1];
          --at;
        }
        order[at] = index;
      }
    return length;
  }
  HOT_DEVICE void Prune(int64_t minimum, int64_t limit) {
    int length = SortedGuests(candidates, -1, -1, true, minimum);
    for (int i = 0; i < length; ++i) {
      int index = candidates[i], target = index / e, expert = index % e, owner = expert / h;
      int64_t rows = copies[index];
      if (!rows || rows >= minimum) continue;
      int64_t needed = Max(loads[owner] + rows - limit, 0);
      int edge_count = SortedGuests(edges, owner, index, false, INT64_MAX);
      for (int j = 0; j < n; ++j) trial[j] = copies[j];
      trial[owner * e + expert] += rows;
      trial[index] = 0;
      for (int k = 0; k < r; ++k) headroom[k] = limit - loads[k] + (k == target ? rows : 0);
      for (int j = 0; j < edge_count && needed; ++j) {
        int edge = edges[j], rank = edge / e, logical = edge % e;
        int64_t moved = Min(needed, Min(trial[owner * e + logical], headroom[rank]));
        if (moved <= 0) continue;
        trial[owner * e + logical] -= moved;
        trial[edge] += moved;
        headroom[rank] -= moved;
        needed -= moved;
      }
      if (!needed) {
        for (int j = 0; j < n; ++j) copies[j] = trial[j];
        Refresh(copies);
      }
    }
  }
  HOT_DEVICE void Rebalance() {
    while (true) {
      int receiver = -1, expert = -1;
      int64_t best = 0, donor_load = 0;
      for (int k = 0; k < r; ++k)
        for (int j = 0; j < e; ++j) {
          int owner = j / h;
          if (owner == k || copies[k * e + j] <= 0) continue;
          int64_t moved = Min(copies[owner * e + j], (loads[owner] - loads[k]) / 2);
          if (moved > best || (moved > 0 && moved == best && loads[owner] > donor_load)) {
            best = moved;
            receiver = k;
            expert = j;
            donor_load = loads[owner];
          }
        }
      if (!best) return;
      int owner = expert / h;
      copies[owner * e + expert] -= best;
      copies[receiver * e + expert] += best;
      loads[owner] -= best;
      loads[receiver] += best;
    }
  }
  HOT_DEVICE int Materialize(HOT_OUTPUT int64_t *control, HOT_OUTPUT int32_t *dispatch, int64_t capacity) {
    int size = r * w;
    HOT_OUTPUT int64_t *slots = control + 1;
    HOT_OUTPUT int64_t *destination = slots + size;
    HOT_OUTPUT int64_t *splits = destination + size;
    HOT_OUTPUT int64_t *order = splits + r * r;
    HOT_OUTPUT int64_t *ordered_counts = order + size;
    int cursor = 0;
    control[0] = 0;
    for (int i = 0; i < r * size; ++i) dispatch[i] = 0;
    for (int i = 0; i < r * r; ++i) splits[i] = 0;
    for (int k = 0; k < r; ++k) {
      int guest = 0;
      for (int j = 0; j < h; ++j) slots[k * w + j] = k * h + j;
      for (int j = h; j < w; ++j) slots[k * w + j] = -1;
      for (int j = 0; j < e; ++j)
        if (j / h != k && copies[k * e + j] > 0) {
          if (guest == b) {
            control[0] = 1;
            return 1;
          }
          slots[k * w + h + guest++] = j;
        }
      int64_t total = 0;
      for (int j = 0; j < w; ++j) {
        int expert = slots[k * w + j];
        destination[k * w + j] = expert < 0 ? 0 : copies[k * e + expert];
        total += destination[k * w + j];
      }
      if (capacity >= 0 && total > capacity) {
        control[0] = 1;
        return 1;
      }
    }
    for (int j = 0; j < e; ++j) {
      // Retain maximal local intersections before distributing remaining occurrences.
      for (int k = 0; k < r; ++k) {
        int slot = -1;
        for (int s = 0; s < w; ++s)
          if (slots[k * w + s] == j) {
            slot = s;
            break;
          }
        targets[k] = slot < 0 ? -1 : k * w + slot;
        if (targets[k] >= 0) order[cursor++] = targets[k];
        int64_t local = Min(counts[k * e + j], copies[k * e + j]);
        if (local > INT32_MAX) {
          control[0] = 1;
          return 1;
        }
        if (local) dispatch[k * size + targets[k]] = local;
        counts[k * e + j] -= local;
        copies[k * e + j] -= local;
      }
      int target = 0;
      for (int source = 0; source < r; ++source) {
        int64_t pending = counts[source * e + j];
        while (pending) {
          while (target < r && !copies[target * e + j]) ++target;
          if (target == r) {
            control[0] = 1;
            return 1;
          }
          int64_t moved = Min(pending, copies[target * e + j]);
          if (moved > INT32_MAX - dispatch[source * size + targets[target]]) {
            control[0] = 1;
            return 1;
          }
          dispatch[source * size + targets[target]] += moved;
          copies[target * e + j] -= moved;
          pending -= moved;
        }
      }
    }
    for (int slot = 0; slot < size; ++slot)
      if (slots[slot] < 0) order[cursor++] = slot;
    for (int source = 0; source < r; ++source)
      for (int i = 0; i < size; ++i) ordered_counts[source * size + i] = dispatch[source * size + order[i]];
    for (int source = 0; source < r; ++source)
      for (int target = 0; target < r; ++target) {
        int64_t total = 0;
        for (int slot = 0; slot < w; ++slot) total += dispatch[source * size + target * w + slot];
        splits[source * r + target] = total;
      }
    return 0;
  }
  int r, e, h, b, n, w;
  HOT_LOCAL int64_t *counts, *copies, *trial, *candidates, *edges;
  HOT_LOCAL int64_t *totals, *remaining, *segments, *loads, *original, *targets, *headroom;
};
}  // namespace hot
