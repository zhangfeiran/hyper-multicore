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
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ReplicaHost {
namespace {

using Matrix = std::vector<std::vector<int64_t>>;
using Entries = std::vector<std::vector<std::pair<int64_t, int64_t>>>;
using Groups = std::vector<std::vector<size_t>>;

struct Geometry {
  size_t peers;
  size_t home;
  size_t width;
  size_t experts;
  size_t physical;
};

struct Quota {
  size_t physical;
  int64_t remaining;
};

void Decref(PyObject *value) { Py_XDECREF(value); }
using PyOwner = std::unique_ptr<PyObject, decltype(&Decref)>;

bool Count(PyObject *value, int64_t *result) {
  if (!PyLong_CheckExact(value)) return false;
  int overflow = 0;
  const PY_LONG_LONG converted = PyLong_AsLongLongAndOverflow(value, &overflow);
  if (PyErr_Occurred() || overflow || converted < 0) return false;
  *result = converted;
  return true;
}

bool Shape(PyObject *counts, PyObject *copies, Py_ssize_t home, Py_ssize_t budget,
           PyObject *nodes, Geometry *shape) {
  if (!PyTuple_CheckExact(counts) || !PyList_CheckExact(copies) || !PyTuple_CheckExact(nodes)) return false;
  const Py_ssize_t peers = PyTuple_GET_SIZE(counts);
  const Py_ssize_t node_count = PyTuple_GET_SIZE(nodes);
  if (peers <= 0 || home <= 0 || budget < 0 || PyList_GET_SIZE(copies) != peers ||
      (node_count != 0 && node_count != peers)) {
    PyErr_SetString(PyExc_ValueError, "Invalid replica host materialization shape");
    return false;
  }
  if (home > PY_SSIZE_T_MAX - budget || home + budget > PY_SSIZE_T_MAX / peers) return false;
  *shape = {static_cast<size_t>(peers), static_cast<size_t>(home), static_cast<size_t>(home + budget),
            static_cast<size_t>(peers * home), static_cast<size_t>(peers * (home + budget))};
  return true;
}

bool AddCount(int64_t value, int64_t *total) {
  if (value > std::numeric_limits<int64_t>::max() - *total) return false;
  *total += value;
  return true;
}

bool ReadCounts(PyObject *counts, const Geometry &shape, Matrix *input, std::vector<int64_t> *totals) {
  // Reject an impossible declared width before creating the native matrices.
  for (size_t source = 0; source < shape.peers; ++source) {
    PyObject *row = PyTuple_GET_ITEM(counts, source);
    if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) != static_cast<Py_ssize_t>(shape.experts)) return false;
  }
  input->assign(shape.peers, std::vector<int64_t>(shape.experts));
  totals->assign(shape.experts, 0);
  for (size_t source = 0; source < shape.peers; ++source) {
    PyObject *row = PyTuple_GET_ITEM(counts, source);
    for (size_t expert = 0; expert < shape.experts; ++expert) {
      int64_t value = 0;
      if (!Count(PyTuple_GET_ITEM(row, expert), &value) || !AddCount(value, &(*totals)[expert])) return false;
      (*input)[source][expert] = value;
    }
  }
  return true;
}

bool ReadCopies(PyObject *copies, const Geometry &shape, const std::vector<int64_t> &expected, Entries *rows) {
  rows->resize(shape.peers);
  std::vector<int64_t> totals(shape.experts, 0);
  for (size_t rank = 0; rank < shape.peers; ++rank) {
    PyObject *row = PyList_GET_ITEM(copies, rank);
    if (!PyDict_CheckExact(row)) return false;
    Py_ssize_t position = 0;
    PyObject *key = nullptr;
    PyObject *value = nullptr;
    while (PyDict_Next(row, &position, &key, &value)) {
      int64_t expert = 0;
      int64_t count = 0;
      if (!Count(key, &expert) || !Count(value, &count) || static_cast<size_t>(expert) >= shape.experts ||
          !AddCount(count, &totals[expert])) return false;
      (*rows)[rank].emplace_back(expert, count);
    }
  }
  if (totals != expected) {
    PyErr_SetString(PyExc_ValueError, "Replica host quotas must conserve every logical expert count");
    return false;
  }
  return true;
}

bool NodeSources(PyObject *nodes, const Geometry &shape, std::vector<size_t> *rank_node, Groups *sources) {
  if (PyTuple_GET_SIZE(nodes) == 0) return true;
  std::vector<int64_t> labels;
  rank_node->resize(shape.peers);
  for (size_t rank = 0; rank < shape.peers; ++rank) {
    int64_t label = 0;
    if (!Count(PyTuple_GET_ITEM(nodes, rank), &label)) return false;
    auto found = std::find(labels.begin(), labels.end(), label);
    const size_t index = static_cast<size_t>(found - labels.begin());
    if (found == labels.end()) {
      labels.push_back(label);
      sources->emplace_back();
    }
    (*rank_node)[rank] = index;
    (*sources)[index].push_back(rank);
  }
  if (labels.size() == 1 || labels.size() == shape.peers) sources->clear();
  return true;
}

bool BuildSlots(const Entries &copies, const Geometry &shape, Matrix *slots,
                std::vector<std::vector<Quota>> *by_expert) {
  slots->assign(shape.peers, std::vector<int64_t>(shape.width, -1));
  by_expert->resize(shape.experts);
  for (size_t rank = 0; rank < shape.peers; ++rank) {
    std::vector<int64_t> guests;
    for (const auto &entry : copies[rank]) {
      if (static_cast<size_t>(entry.first) / shape.home != rank && entry.second) guests.push_back(entry.first);
    }
    std::sort(guests.begin(), guests.end());
    if (guests.size() > shape.width - shape.home) {
      PyErr_SetString(PyExc_ValueError, "replica layout exceeds the physical guest slot cap");
      return false;
    }
    for (size_t slot = 0; slot < shape.home; ++slot) (*slots)[rank][slot] = rank * shape.home + slot;
    std::copy(guests.begin(), guests.end(), (*slots)[rank].begin() + shape.home);
    for (size_t slot = 0; slot < shape.width; ++slot) {
      const int64_t expert = (*slots)[rank][slot];
      if (expert < 0) continue;
      for (const auto &entry : copies[rank]) {
        if (entry.first == expert && entry.second) {
          (*by_expert)[expert].push_back({rank * shape.width + slot, entry.second});
          break;
        }
      }
    }
  }
  return true;
}

void FillNodeQuotas(std::vector<Quota> *quotas, std::vector<int64_t> *pending, Matrix *dispatch,
                    const Geometry &shape, const std::vector<size_t> &rank_node, const Groups &sources) {
  Groups destinations(sources.size());
  std::vector<size_t> order;
  for (size_t index = 0; index < quotas->size(); ++index) {
    const auto &quota = (*quotas)[index];
    if (quota.remaining == 0) continue;
    const size_t node = rank_node[quota.physical / shape.width];
    if (destinations[node].empty()) order.push_back(node);
    destinations[node].push_back(index);
  }
  for (size_t node : order) {
    size_t index = 0;
    for (size_t source : sources[node]) {
      while ((*pending)[source] && index < destinations[node].size()) {
        auto &quota = (*quotas)[destinations[node][index]];
        const int64_t moved = std::min((*pending)[source], quota.remaining);
        (*dispatch)[source][quota.physical] += moved;
        (*pending)[source] -= moved;
        quota.remaining -= moved;
        if (!quota.remaining) ++index;
      }
    }
  }
}

void AssignExpert(size_t expert, std::vector<Quota> *quotas, const Matrix &input, Matrix *dispatch,
                  const Geometry &shape, const std::vector<size_t> &rank_node, const Groups &sources) {
  const size_t owner = expert / shape.home * shape.width + expert % shape.home;
  if (quotas->empty() || (quotas->size() == 1 && (*quotas)[0].physical == owner)) return;
  for (auto &row : *dispatch) row[owner] = 0;
  if (quotas->size() == 1) {
    for (size_t source = 0; source < shape.peers; ++source) {
      (*dispatch)[source][(*quotas)[0].physical] = input[source][expert];
    }
    return;
  }
  std::vector<int64_t> pending(shape.peers);
  for (size_t source = 0; source < shape.peers; ++source) pending[source] = input[source][expert];
  for (auto &quota : *quotas) {
    const size_t source = quota.physical / shape.width;
    const int64_t moved = std::min(pending[source], quota.remaining);
    (*dispatch)[source][quota.physical] = moved;
    pending[source] -= moved;
    quota.remaining -= moved;
  }
  if (!sources.empty()) FillNodeQuotas(quotas, &pending, dispatch, shape, rank_node, sources);
  size_t index = 0;
  for (size_t source = 0; source < shape.peers; ++source) {
    while (pending[source]) {
      while ((*quotas)[index].remaining == 0) ++index;
      auto &quota = (*quotas)[index];
      const int64_t moved = std::min(pending[source], quota.remaining);
      (*dispatch)[source][quota.physical] += moved;
      pending[source] -= moved;
      quota.remaining -= moved;
    }
  }
}

PyObject *Encode(const Matrix &matrix, const Geometry &shape, PyObject *counts = nullptr,
                 const Matrix *input = nullptr) {
  PyOwner result(PyTuple_New(matrix.size()), &Decref);
  if (result == nullptr) return nullptr;
  for (size_t source = 0; source < matrix.size(); ++source) {
    PyOwner row(PyTuple_New(matrix[source].size()), &Decref);
    if (row == nullptr) return nullptr;
    for (size_t slot = 0; slot < matrix[source].size(); ++slot) {
      PyObject *value = nullptr;
      const size_t local = slot % shape.width;
      const size_t expert = slot / shape.width * shape.home + local;
      if (counts != nullptr && local < shape.home && matrix[source][slot] == (*input)[source][expert]) {
        value = PyTuple_GET_ITEM(PyTuple_GET_ITEM(counts, source), expert);
        Py_INCREF(value);
      } else {
        value = PyLong_FromLongLong(matrix[source][slot]);
      }
      if (value == nullptr) return nullptr;
      PyTuple_SET_ITEM(row.get(), slot, value);
    }
    PyTuple_SET_ITEM(result.get(), source, row.release());
  }
  return result.release();
}

PyObject *Materialize(PyObject *counts, PyObject *copies, Py_ssize_t home, Py_ssize_t budget, PyObject *nodes) {
  Geometry shape{};
  Matrix input;
  Entries rows;
  std::vector<int64_t> totals;
  std::vector<size_t> rank_node;
  Groups sources;
  if (!Shape(counts, copies, home, budget, nodes, &shape) || !ReadCounts(counts, shape, &input, &totals) ||
      !ReadCopies(copies, shape, totals, &rows) || !NodeSources(nodes, shape, &rank_node, &sources)) {
    if (PyErr_Occurred()) return nullptr;
    Py_RETURN_NONE;
  }
  Matrix slots;
  std::vector<std::vector<Quota>> quotas;
  if (!BuildSlots(rows, shape, &slots, &quotas)) return nullptr;
  Matrix dispatch(shape.peers, std::vector<int64_t>(shape.physical, 0));
  for (size_t source = 0; source < shape.peers; ++source) {
    for (size_t expert = 0; expert < shape.experts; ++expert) {
      dispatch[source][expert / shape.home * shape.width + expert % shape.home] = input[source][expert];
    }
  }
  for (size_t expert = 0; expert < shape.experts; ++expert) {
    AssignExpert(expert, &quotas[expert], input, &dispatch, shape, rank_node, sources);
  }
  PyOwner slot_result(Encode(slots, shape), &Decref);
  if (slot_result == nullptr) return nullptr;
  PyOwner dispatch_result(Encode(dispatch, shape, counts, &input), &Decref);
  if (dispatch_result == nullptr) return nullptr;
  return PyTuple_Pack(2, slot_result.get(), dispatch_result.get());
}

}  // namespace

PyObject *MaterializeDispatch(PyObject *, PyObject *arguments) {
  PyObject *counts = nullptr;
  PyObject *copies = nullptr;
  PyObject *nodes = nullptr;
  Py_ssize_t home = 0;
  Py_ssize_t budget = 0;
  if (!PyArg_ParseTuple(arguments, "OOnnO", &counts, &copies, &home, &budget, &nodes)) return nullptr;
  try {
    return Materialize(counts, copies, home, budget, nodes);
  } catch (const std::bad_alloc &) {
    return PyErr_NoMemory();
  } catch (const std::length_error &) {
    PyErr_SetString(PyExc_OverflowError, "Replica host materialization exceeds addressable memory");
    return nullptr;
  }
}

}  // namespace ReplicaHost
