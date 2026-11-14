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

PyObject *AbiVersion(PyObject *, PyObject *) { return PyLong_FromLong(3); }

PyMethodDef methods[] = {
  {"abi_version", AbiVersion, METH_NOARGS, "Return the host primitive ABI version."},
  {"validate_counts", ValidateCounts, METH_O, "Validate a frozen ordinary integer matrix."},
  {"column_totals", ColumnTotals, METH_O, "Sum ordinary tuple/list integer columns within the native signed range."},
  {"materialize_dispatch", ReplicaHost::MaterializeDispatch, METH_VARARGS,
   "Materialize exact physical slots and source dispatch quotas."},
  {"locality_moves", ReplicaHost::LocalityMoves, METH_VARARGS,
   "Return exact whole-guest placement operations without changing Python input objects."},
  {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {PyModuleDef_HEAD_INIT, "_replica_host", nullptr, -1, methods, nullptr, nullptr, nullptr, nullptr};

}  // namespace

PyMODINIT_FUNC PyInit__replica_host() { return PyModule_Create(&module); }
