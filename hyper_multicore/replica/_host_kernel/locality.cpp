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
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <new>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace ReplicaHost {
namespace {

using Row = std::map<int64_t, int64_t>;
using Move = std::array<Py_ssize_t, 4>;
using Score = std::tuple<int64_t, int64_t, int64_t, int64_t>;

bool Count(PyObject *value, int64_t *result) {
  if (!PyLong_CheckExact(value)) return false;
  int overflow = 0;
  const auto converted = PyLong_AsLongLongAndOverflow(value, &overflow);
  if (PyErr_Occurred() || overflow || converted < 0) return false;
  *result = converted;
  return true;
}

struct Placement {
  size_t home = 0;
  int64_t budget = 0;
  int64_t ceiling = 0;
  std::vector<Row> copies;
  std::vector<int64_t> loads;
  std::vector<size_t> guests;
  std::vector<size_t> nodes;
  std::vector<std::vector<size_t>> groups;
  std::vector<Move> moves;
};

bool ReadRow(PyObject *copies, Py_ssize_t rank, Py_ssize_t peers, int64_t home_value, Placement *state) {
    PyObject *row = PyList_GET_ITEM(copies, rank);
    if (!PyDict_CheckExact(row)) return false;
    Py_ssize_t position = 0;
    PyObject *key = nullptr;
    PyObject *value = nullptr;
    while (PyDict_Next(row, &position, &key, &value)) {
      int64_t expert = 0;
      int64_t rows = 0;
      if (!Count(key, &expert) || expert >= peers * home_value || !Count(value, &rows) ||
          rows > std::numeric_limits<int64_t>::max() - state->loads[rank]) return false;
      state->copies[rank].emplace(expert, rows);
      state->loads[rank] += rows;
      state->guests[rank] += rows > 0 && expert / home_value != rank;
    }
    state->ceiling = std::max(state->ceiling, state->loads[rank]);
  return true;
}

bool Read(PyObject *copies, PyObject *home, PyObject *budget, PyObject *nodes, Placement *state) {
  int64_t home_value = 0;
  if (!PyList_CheckExact(copies) || !PyTuple_CheckExact(nodes) || !Count(home, &home_value) ||
      home_value <= 0 || !Count(budget, &state->budget)) return false;
  const auto peers = PyList_GET_SIZE(copies);
  if (peers <= 0 || PyTuple_GET_SIZE(nodes) != peers || home_value > PY_SSIZE_T_MAX / peers) return false;
  state->home = static_cast<size_t>(home_value);
  state->copies.resize(peers);
  state->loads.resize(peers, 0);
  state->guests.resize(peers, 0);
  state->nodes.resize(peers);
  std::map<int64_t, size_t> labels;
  for (Py_ssize_t rank = 0; rank < peers; ++rank) {
    int64_t label = 0;
    if (!Count(PyTuple_GET_ITEM(nodes, rank), &label)) return false;
    const auto found = labels.find(label);
    if (found == labels.end()) {
      labels.emplace(label, state->groups.size());
      state->groups.emplace_back();
    }
    state->nodes[rank] = labels.at(label);
    state->groups[state->nodes[rank]].push_back(rank);
    if (!ReadRow(copies, rank, peers, home_value, state)) return false;
  }
  return true;
}

bool MoveLocal(Placement *state, size_t source, int64_t expert, int64_t rows) {
  const size_t owner = static_cast<size_t>(expert) / state->home;
  auto targets = state->groups[state->nodes[owner]];
  std::sort(targets.begin(), targets.end(), [state, expert](size_t left, size_t right) {
    return std::make_tuple(state->copies[left].count(expert) == 0, state->loads[left], left) <
           std::make_tuple(state->copies[right].count(expert) == 0, state->loads[right], right);
  });
  for (const size_t target : targets) {
    const bool present = state->copies[target].count(expert) != 0;
    if (rows > state->ceiling - state->loads[target] ||
        (!present && state->guests[target] >= static_cast<size_t>(state->budget))) continue;
    if (!present && owner != target) ++state->guests[target];
    state->copies[target][expert] += rows;
    state->copies[source].erase(expert);
    --state->guests[source];
    state->loads[source] -= rows;
    state->loads[target] += rows;
    state->moves.push_back({static_cast<Py_ssize_t>(source), static_cast<Py_ssize_t>(target),
                            static_cast<Py_ssize_t>(expert), -1});
    return true;
  }
  return false;
}

void SwapLocal(Placement *state, size_t source, int64_t expert, int64_t rows) {
  const size_t owner = static_cast<size_t>(expert) / state->home;
  Score best{0, 0, 0, 0};
  size_t best_target = 0;
  int64_t best_expert = -1;
  int64_t best_amount = 0;
  for (const size_t target : state->groups[state->nodes[owner]]) {
    if (state->copies[target].count(expert)) continue;
    for (const auto &entry : state->copies[target]) {
      const auto other = entry.first;
      const auto amount = entry.second;
      const auto other_owner = static_cast<size_t>(other) / state->home;
      if (!amount || other_owner == target || state->copies[source].count(other)) continue;
      // These equivalent headroom inequalities avoid signed overflow before
      // constructing the same Python reference score.
      const auto lower = rows - (state->ceiling - state->loads[target]);
      const auto upper = rows + (state->ceiling - state->loads[source]);
      if (amount < lower || amount > upper) continue;
      const int64_t saved = 1 + static_cast<int64_t>(state->nodes[other_owner] != state->nodes[target]) -
                            static_cast<int64_t>(state->nodes[other_owner] != state->nodes[source]);
      const auto left = state->loads[source] - rows + amount;
      const auto right = state->loads[target] - amount + rows;
      const Score score{saved, -std::max(left, right), -static_cast<int64_t>(target), -other};
      if (saved > 0 && score > best) {
        best = score;
        best_target = target;
        best_expert = other;
        best_amount = amount;
      }
    }
  }
  if (best_expert < 0) return;
  state->copies[source].erase(expert);
  state->copies[best_target].erase(best_expert);
  state->copies[source][best_expert] = best_amount;
  state->copies[best_target][expert] = rows;
  state->loads[source] = state->loads[source] - rows + best_amount;
  state->loads[best_target] = state->loads[best_target] - best_amount + rows;
  state->moves.push_back({static_cast<Py_ssize_t>(source), static_cast<Py_ssize_t>(best_target),
                          static_cast<Py_ssize_t>(expert), static_cast<Py_ssize_t>(best_expert)});
}

PyObject *Search(Placement *state) {
  std::vector<std::tuple<int64_t, size_t, int64_t>> edges;
  for (size_t rank = 0; rank < state->copies.size(); ++rank) {
    for (const auto &entry : state->copies[rank]) {
      const auto expert = entry.first;
      const auto rows = entry.second;
      const auto owner = static_cast<size_t>(expert) / state->home;
      if (rows && owner != rank && state->nodes[owner] != state->nodes[rank]) {
        edges.emplace_back(-rows, rank, expert);
      }
    }
  }
  std::sort(edges.begin(), edges.end());
  for (const auto &edge : edges) {
    const auto rows = -std::get<0>(edge);
    const auto source = std::get<1>(edge);
    const auto expert = std::get<2>(edge);
    const auto found = state->copies[source].find(expert);
    if (found == state->copies[source].end() || found->second != rows) continue;
    if (!MoveLocal(state, source, expert, rows)) SwapLocal(state, source, expert, rows);
  }
  PyObject *result = PyTuple_New(static_cast<Py_ssize_t>(state->moves.size()));
  if (result == nullptr) return nullptr;
  for (size_t index = 0; index < state->moves.size(); ++index) {
    const auto &move = state->moves[index];
    PyObject *entry = Py_BuildValue("(nnnn)", move[0], move[1], move[2], move[3]);
    if (entry == nullptr) {
      Py_DECREF(result);
      return nullptr;
    }
    PyTuple_SET_ITEM(result, static_cast<Py_ssize_t>(index), entry);
  }
  return result;
}

}  // namespace

PyObject *LocalityMoves(PyObject *, PyObject *arguments) {
  PyObject *copies = nullptr;
  PyObject *home = nullptr;
  PyObject *budget = nullptr;
  PyObject *nodes = nullptr;
  if (!PyArg_ParseTuple(arguments, "OOOO", &copies, &home, &budget, &nodes)) return nullptr;
  try {
    Placement state;
    if (!Read(copies, home, budget, nodes, &state)) {
      if (PyErr_Occurred()) return nullptr;
      Py_RETURN_NONE;
    }
    return Search(&state);
  } catch (const std::bad_alloc &) {
    return PyErr_NoMemory();
  } catch (const std::length_error &) {
    PyErr_SetString(PyExc_OverflowError, "Replica host placement exceeds addressable memory");
    return nullptr;
  }
}

}  // namespace ReplicaHost
