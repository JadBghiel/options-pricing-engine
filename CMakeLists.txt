# c++ core of the engine
#
# 2 ways in:
# - pip / uv (scikit-build-core sets SKBUILD) -> builds the python module ope._core only
# - plain cmake -> the library plus googletest and google benchmark, e.g.
#     cmake -S . -B build/cpp -G Ninja -DOPE_BUILD_TESTS=ON && cmake --build build/cpp && ctest --test-dir build/cpp
cmake_minimum_required(VERSION 3.20)
project(ope_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # for clangd in the editor

# a debug build is 5 to 10x slower -> benchmarking 1 by accident is easy, release unless asked
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "build type" FORCE)
endif()

option(OPE_BUILD_TESTS "googletest unit tests" OFF)
option(OPE_BUILD_BENCHMARKS "google benchmark" OFF)
option(OPE_WERROR "warnings are errors (ci)" OFF)
set(OPE_SANITIZE "" CACHE STRING "sanitizers, e.g. address,undefined or thread")

if(OPE_SANITIZE)
  # on everything, googletest included -> a bad access inside an EXPECT is caught too
  add_compile_options(-fsanitize=${OPE_SANITIZE} -fno-omit-frame-pointer -fno-sanitize-recover=all)
  add_link_options(-fsanitize=${OPE_SANITIZE})
endif()

find_package(Threads REQUIRED)

add_library(ope_core STATIC
  cpp/src/black_scholes.cpp
  cpp/src/monte_carlo.cpp
  cpp/src/tree.cpp
)
target_include_directories(ope_core PUBLIC cpp/include)
target_link_libraries(ope_core PUBLIC Threads::Threads)
# the static lib ends up inside a shared python module -> it must be position independent
set_target_properties(ope_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(ope_core PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wconversion)
if(OPE_WERROR)
  target_compile_options(ope_core PRIVATE -Werror)
endif()
# no fused multiply add: a*b + c rounded once instead of twice would change results between
# machines, off is the default on x86-64 w/o -march, this makes it explicit everywhere
target_compile_options(ope_core PUBLIC -ffp-contract=off)

if(SKBUILD)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_core MODULE cpp/python/module.cpp)
  target_link_libraries(_core PRIVATE ope_core)
  target_compile_definitions(_core PRIVATE
    OPE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    OPE_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
  )
  install(TARGETS _core LIBRARY DESTINATION ope)
endif()

include(FetchContent)

if(OPE_BUILD_TESTS)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
    URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
    DOWNLOAD_EXTRACT_TIMESTAMP ON
  )
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_executable(ope_tests
    cpp/tests/test_black_scholes.cpp
    cpp/tests/test_monte_carlo.cpp
    cpp/tests/test_tree.cpp
  )
  target_link_libraries(ope_tests PRIVATE ope_core GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(ope_tests)
endif()

if(OPE_BUILD_BENCHMARKS)
  FetchContent_Declare(benchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.9.5.tar.gz
    URL_HASH SHA256=9631341c82bac4a288bef951f8b26b41f69021794184ece969f8473977eaa340
    DOWNLOAD_EXTRACT_TIMESTAMP ON
  )
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(benchmark)

  add_executable(ope_bench cpp/bench/bench_core.cpp)
  target_link_libraries(ope_bench PRIVATE ope_core benchmark::benchmark)
endif()
