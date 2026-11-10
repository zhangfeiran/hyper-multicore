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
#include <new>
#include <stdexcept>
#include <vector>

namespace ReplicaHost {
PyObject *MaterializeDispatch(PyObject *, PyObject *);
PyObject *LocalityMoves(PyObject *, PyObject *);
}

namespace {

class Owned {
 public:
  explicit Owned(PyObject *value) : value_(value) {}
  ~Owned() { Py_XDECREF(value_); }
  Owned(const Owned &) = delete;
  Owned &operator=(const Owned &) = delete;
  PyObject *Get() const { return value_; }

 private:
  PyObject *value_;
};

bool Integer(PyObject *value, int64_t *result) {
  if (!PyLong_CheckExact(value)) return false;
  int overflow = 0;
  const PY_LONG_LONG converted = PyLong_AsLongLongAndOverflow(value, &overflow);
  if (PyErr_Occurred()) return false;
  if (overflow || converted < 0) return false;
  *result = converted;
  return true;
}

PyObject *ValidateCounts(PyObject *, PyObject *rows) {
  if (!PyTuple_Check(rows) || PyTuple_GET_SIZE(rows) == 0) Py_RETURN_FALSE;
  PyObject *first = PyTuple_GET_ITEM(rows, 0);
  if (!PyTuple_Check(first) || PyTuple_GET_SIZE(first) == 0) Py_RETURN_FALSE;
  const Py_ssize_t width = PyTuple_GET_SIZE(first);
  for (Py_ssize_t rank = 0; rank < PyTuple_GET_SIZE(rows); ++rank) {
    PyObject *row = PyTuple_GET_ITEM(rows, rank);
    if (!PyTuple_Check(row) || PyTuple_GET_SIZE(row) != width) Py_RETURN_FALSE;
    for (Py_ssize_t expert = 0; expert < width; ++expert) {
      int64_t count = 0;
      if (!Integer(PyTuple_GET_ITEM(row, expert), &count)) {
        if (PyErr_Occurred()) return nullptr;
        Py_RETURN_FALSE;
      }
    }
  }
  Py_RETURN_TRUE;
}

bool ExactSequence(PyObject *value) { return PyTuple_CheckExact(value) || PyList_CheckExact(value); }

bool ColumnShape(PyObject *rows, Py_ssize_t *width) {
  if (!ExactSequence(rows) || PySequence_Fast_GET_SIZE(rows) == 0) return false;
  PyObject *first = PySequence_Fast_GET_ITEM(rows, 0);
  if (!ExactSequence(first) || PySequence_Fast_GET_SIZE(first) == 0) return false;
  *width = PySequence_Fast_GET_SIZE(first);
  // Reject every malformed row before allocating even the bounded native vector.
  for (Py_ssize_t rank = 0; rank < PySequence_Fast_GET_SIZE(rows); ++rank) {
    PyObject *row = PySequence_Fast_GET_ITEM(rows, rank);
    if (!ExactSequence(row) || PySequence_Fast_GET_SIZE(row) != *width) return false;
  }
  return true;
}

bool SumColumns(PyObject *rows, std::vector<int64_t> *totals) {
  for (Py_ssize_t rank = 0; rank < PySequence_Fast_GET_SIZE(rows); ++rank) {
    PyObject *row = PySequence_Fast_GET_ITEM(rows, rank);
    for (size_t column = 0; column < totals->size(); ++column) {
      int64_t count = 0;
      if (!Integer(PySequence_Fast_GET_ITEM(row, column), &count) ||
          count > std::numeric_limits<int64_t>::max() - (*totals)[column]) return false;
      (*totals)[column] += count;
    }
  }
  return true;
}

PyObject *ColumnTotals(PyObject *, PyObject *rows) {
  Py_ssize_t width = 0;
  if (!ColumnShape(rows, &width)) Py_RETURN_NONE;
  try {
    std::vector<int64_t> totals(static_cast<size_t>(width), 0);
    if (!SumColumns(rows, &totals)) {
      if (PyErr_Occurred()) return nullptr;
      Py_RETURN_NONE;
    }
    PyObject *result = PyList_New(width);
    if (result == nullptr) return nullptr;
    for (Py_ssize_t column = 0; column < width; ++column) {
      PyObject *value = PyLong_FromLongLong(static_cast<PY_LONG_LONG>(totals[column]));
      if (value == nullptr) {
        Py_DECREF(result);
        return nullptr;
      }
      PyList_SET_ITEM(result, column, value);
    }
    return result;
  } catch (const std::bad_alloc &) {
    return PyErr_NoMemory();
  } catch (const std::length_error &) {
    return PyErr_NoMemory();
  }
}

bool Point(PyObject *table, Py_ssize_t index, int64_t *rows, double *duration) {
  PyObject *point = PyTuple_GET_ITEM(table, index);
  if (!PyTuple_Check(point) || PyTuple_GET_SIZE(point) != 2 || !Integer(PyTuple_GET_ITEM(point, 0), rows)) {
    return false;
  }
  PyObject *value = PyTuple_GET_ITEM(point, 1);
  if (PyLong_CheckExact(value)) {
    int64_t integer = 0;
    // Python retains exact integer multiplication until division for integer
    // time knots. Only zero is safe to promote before that operation.
    if (!Integer(value, &integer) || integer != 0) return false;
  } else if (!PyFloat_CheckExact(value)) {
    return false;
  }
  *duration = PyFloat_AsDouble(value);
  return !PyErr_Occurred();
}

bool Interpolate(PyObject *table, int64_t rows, double *result,
                 const char *range_error = "Expert rows exceed the calibrated cost range") {
  Py_ssize_t left = 0;
  Py_ssize_t right = PyTuple_GET_SIZE(table);
  while (left < right) {
    const Py_ssize_t middle = left + (right - left) / 2;
    int64_t knot = 0;
    double duration = 0;
    if (!Point(table, middle, &knot, &duration)) return false;
    if (knot < rows) {
      left = middle + 1;
    } else {
      right = middle;
    }
  }
  if (left == PyTuple_GET_SIZE(table)) {
    PyErr_SetString(PyExc_ValueError, range_error);
    return false;
  }
  int64_t right_rows = 0;
  double right_ms = 0;
  if (!Point(table, left, &right_rows, &right_ms)) return false;
  if (right_rows == rows) {
    *result = right_ms;
    return true;
  }
  int64_t left_rows = 0;
  double left_ms = 0;
  if (left == 0 || !Point(table, left - 1, &left_rows, &left_ms) || right_rows <= left_rows) return false;
  // Match Python's left-associated multiply/divide before the final addition.
  const double product = (right_ms - left_ms) * static_cast<double>(rows - left_rows);
  *result = left_ms + product / static_cast<double>(right_rows - left_rows);
  return true;
}

bool Duration(PyObject *cache, PyObject *rows_object, int64_t rows, PyObject *forward, PyObject *backward,
              std::array<double, 2> *result) {
  PyObject *cached = PyDict_GetItemWithError(cache, rows_object);
  if (cached == nullptr && PyErr_Occurred()) return false;
  if (cached != nullptr) {
    if (!PyTuple_CheckExact(cached) || PyTuple_GET_SIZE(cached) != 2) return false;
    for (Py_ssize_t phase = 0; phase < 2; ++phase) {
      PyObject *value = PyTuple_GET_ITEM(cached, phase);
      if (!PyFloat_CheckExact(value) && !PyLong_CheckExact(value)) return false;
      (*result)[phase] = PyFloat_AsDouble(value);
      if (PyErr_Occurred()) return false;
    }
    return true;
  }
  if (!Interpolate(forward, rows, &(*result)[0]) || !Interpolate(backward, rows, &(*result)[1])) return false;
  Owned pair(Py_BuildValue("(dd)", (*result)[0], (*result)[1]));
  return pair.Get() != nullptr && PyDict_SetItem(cache, rows_object, pair.Get()) == 0;
}

bool AddLocalRows(PyObject *source, int64_t expert, int64_t rows, int64_t *local) {
  if (source == Py_None) return true;
  int64_t supplied = 0;
  if (expert >= PyTuple_GET_SIZE(source) || !Integer(PyTuple_GET_ITEM(source, expert), &supplied)) return false;
  const int64_t amount = std::min(supplied, rows);
  if (amount > std::numeric_limits<int64_t>::max() - *local) return false;
  *local += amount;
  return true;
}

PyObject *RankCompute(PyObject *, PyObject *arguments) {
  PyObject *experts = nullptr;
  PyObject *source = nullptr;
  PyObject *cache = nullptr;
  PyObject *forward = nullptr;
  PyObject *backward = nullptr;
  int home = 0;
  int rank = 0;
  if (!PyArg_ParseTuple(arguments, "OOiiOOO", &experts, &source, &home, &rank, &cache, &forward, &backward)) {
    return nullptr;
  }
  // Dict subclasses can change items()/get() semantics without changing the
  // underlying PyDict_Next order, so the Python reference must handle them.
  if (!PyDict_CheckExact(experts) || !PyDict_CheckExact(cache)) Py_RETURN_NONE;
  if (!PyTuple_Check(forward) || !PyTuple_Check(backward) || PyTuple_GET_SIZE(forward) < 2 ||
      PyTuple_GET_SIZE(backward) < 2 || (source != Py_None && !PyTuple_Check(source)) || home <= 0 || rank < 0) {
    PyErr_SetString(PyExc_ValueError, "Invalid replica host compute inputs");
    return nullptr;
  }
  std::array<double, 2> forward_bins = {0.0, 0.0};
  std::array<double, 2> backward_bins = {0.0, 0.0};
  int64_t local = 0;
  Py_ssize_t position = 0;
  PyObject *expert_object = nullptr;
  PyObject *rows_object = nullptr;
  // PyDict_Next retains the exact insertion order used by the reference.
  while (PyDict_Next(experts, &position, &expert_object, &rows_object)) {
    int64_t expert = 0;
    int64_t rows = 0;
    if (!Integer(expert_object, &expert) || !Integer(rows_object, &rows)) {
      if (PyErr_Occurred()) return nullptr;
      Py_RETURN_NONE;
    }
    std::array<double, 2> durations;
    if (!Duration(cache, rows_object, rows, forward, backward, &durations)) {
      if (PyErr_Occurred()) return nullptr;
      Py_RETURN_NONE;
    }
    const int guest = expert / home != rank;
    forward_bins[guest] += durations[0];
    backward_bins[guest] += durations[1];
    if (!AddLocalRows(source, expert, rows, &local)) {
      if (PyErr_Occurred()) return nullptr;
      Py_RETURN_NONE;
    }
  }
  return Py_BuildValue("(ddL)", forward_bins[0] + forward_bins[1], backward_bins[0] + backward_bins[1],
                       static_cast<PY_LONG_LONG>(local));
}

bool OrderedPhase(PyObject *experts, int home, int rank, PyObject *compute, PyObject *owner, bool enabled,
                  PyObject *guests, double *home_ms) {
  Py_ssize_t position = 0;
  PyObject *expert_object = nullptr;
  PyObject *rows_object = nullptr;
  while (PyDict_Next(experts, &position, &expert_object, &rows_object)) {
    int64_t expert = 0;
    int64_t rows = 0;
    if (!Integer(expert_object, &expert) || !Integer(rows_object, &rows)) return false;
    if (expert / home == rank) {
      double duration = 0.0;
      const char *error = enabled ? "Expert rows exceed the calibrated cost range"
                                  : "Owner rows exceed the critical-path calibration range";
      if (!Interpolate(enabled ? compute : owner, rows, &duration, error)) return false;
      *home_ms += duration;
    } else if (rows) {
      double duration = 0.0;
      if (!Interpolate(compute, rows, &duration)) return false;
      Owned guest(Py_BuildValue("(Ld)", static_cast<PY_LONG_LONG>(expert), duration));
      if (guest.Get() == nullptr || PyList_Append(guests, guest.Get()) != 0) return false;
    }
  }
  return true;
}

PyObject *OrderedCompute(PyObject *, PyObject *arguments) {
  PyObject *experts = nullptr;
  PyObject *compute = nullptr;
  PyObject *owner = nullptr;
  int home = 0;
  int rank = 0;
  int enabled = 0;
  if (!PyArg_ParseTuple(arguments, "OiiOOp", &experts, &home, &rank, &compute, &owner, &enabled)) {
    return nullptr;
  }
  if (!PyDict_CheckExact(experts)) Py_RETURN_NONE;
  if (!PyTuple_Check(compute) || !PyTuple_Check(owner) || PyTuple_GET_SIZE(compute) < 2 ||
      PyTuple_GET_SIZE(owner) < 2 || home <= 0 || rank < 0) {
    PyErr_SetString(PyExc_ValueError, "Invalid replica host phase inputs");
    return nullptr;
  }
  Owned guests(PyList_New(0));
  if (guests.Get() == nullptr) return nullptr;
  double home_ms = 0.0;
  if (!OrderedPhase(experts, home, rank, compute, owner, enabled, guests.Get(), &home_ms)) {
    if (PyErr_Occurred()) return nullptr;
    Py_RETURN_NONE;
  }
  return Py_BuildValue("(dN)", home_ms, PyList_AsTuple(guests.Get()));
}

PyObject *AbiVersion(PyObject *, PyObject *) { return PyLong_FromLong(3); }

PyMethodDef methods[] = {
  {"abi_version", AbiVersion, METH_NOARGS, "Return the host primitive ABI version."},
  {"validate_counts", ValidateCounts, METH_O, "Validate a frozen ordinary integer matrix."},
  {"column_totals", ColumnTotals, METH_O, "Sum ordinary tuple/list integer columns within the native signed range."},
  {"rank_compute", RankCompute, METH_VARARGS, "Accumulate exact ordered calibrated expert compute bins."},
  {"ordered_compute", OrderedCompute, METH_VARARGS, "Interpolate ordered guest rows and serially sum home rows."},
  {"materialize_dispatch", ReplicaHost::MaterializeDispatch, METH_VARARGS,
   "Materialize exact physical slots and source dispatch quotas."},
  {"locality_moves", ReplicaHost::LocalityMoves, METH_VARARGS,
   "Return exact whole-guest placement operations without changing Python input objects."},
  {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {PyModuleDef_HEAD_INIT, "_replica_host", nullptr, -1, methods, nullptr, nullptr, nullptr, nullptr};

}  // namespace

PyMODINIT_FUNC PyInit__replica_host() { return PyModule_Create(&module); }
