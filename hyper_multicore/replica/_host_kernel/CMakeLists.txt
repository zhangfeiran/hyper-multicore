# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
cmake_minimum_required(VERSION 3.18)
project(hyper_multicore_replica_host LANGUAGES CXX)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
python3_add_library(_replica_host MODULE WITH_SOABI host.cpp dispatch.cpp locality.cpp)
target_compile_features(_replica_host PRIVATE cxx_std_17)
target_compile_options(_replica_host PRIVATE
    -O3 -fno-fast-math -ffp-contract=off -Wall -Wextra)
install(TARGETS _replica_host LIBRARY DESTINATION .)
