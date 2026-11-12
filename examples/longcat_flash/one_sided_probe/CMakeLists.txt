# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

cmake_minimum_required(VERSION 3.18)
set(CMAKE_C_COMPILER bisheng)
set(CMAKE_CXX_COMPILER bisheng)
project(longcat_one_sided_probe LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT DEFINED ENV{ASCEND_HOME_PATH} OR NOT EXISTS "${SHMEM_SDK_ROOT}/include/shmem.h")
  message(FATAL_ERROR "Activate CANN and specify SHMEM_SDK_ROOT with an RDMA-enabled SDK")
endif()
set(cann "$ENV{ASCEND_HOME_PATH}")
include_directories("${cann}/include" "${cann}/compiler/tikcpp" "${cann}/compiler/tikcpp/tikcfw"
                    "${cann}/compiler/tikcpp/tikcfw/impl" "${cann}/compiler/tikcpp/tikcfw/interface"
                    "${SHMEM_SDK_ROOT}/include" "${SHMEM_SDK_ROOT}/src/device")
link_directories("${cann}/lib64" "${SHMEM_SDK_ROOT}/lib")
add_library(probe_kernel OBJECT kernel.cpp)
target_compile_options(probe_kernel PRIVATE -O2 -xcce --cce-aicore-arch=dav-c220-vec
  "SHELL:-mllvm -cce-aicore-stack-size=0x8000" "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
  "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false")
add_library(probe_host OBJECT host.cpp)
target_compile_options(probe_host PRIVATE -O2 -xc++)
add_library(longcat_one_sided_probe SHARED $<TARGET_OBJECTS:probe_kernel> $<TARGET_OBJECTS:probe_host>)
target_link_options(longcat_one_sided_probe PRIVATE --cce-fatobj-link)
target_link_libraries(longcat_one_sided_probe PRIVATE hyper_multicore_shmem ascendcl runtime stdc++ m dl)
install(TARGETS longcat_one_sided_probe LIBRARY DESTINATION lib)
