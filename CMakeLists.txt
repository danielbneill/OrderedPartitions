cmake_minimum_required(VERSION 3.6)

project(proto)

option(USE_C++17 "Use C++17 compliant compiler, default is C++11" OFF)
option(USE_C++14 "Use C++14 compliant compiler, default is C++!1" OFF)

if(CMAKE_CXX_COMPILER_LOADED)
  message(STATUS "The C++ compiler ID is: ${CMAKE_CXX_COMPILER_ID}")
  message(STATUS "The C++ compiler version is: ${CMAKE_CXX_COMPILER_VERSION}")
endif()

if (USE_C++17)
  set(CMAKE_CXX_STANDARD 17)
  message(STATUS "C++17-compliant CXX")
elseif (USE_C++14)
  set(CMAKE_CXX_STANDARD 14)
  message(STATUS "C++14-compliant CXX")
else()
  set(CMAKE_CXX_STANDARD 11)
  message(STATUS "C++11-compliant CXX")
endif()

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
cmake_policy(SET CMP0078 OLD)
cmake_policy(SET CMP0086 OLD)

include(GNUInstallDirs)

# GTest
find_package(GTest REQUIRED)

# SWIG, Python
find_package(SWIG REQUIRED)
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)

# Boost
find_package(Boost REQUIRED)

include(UseSWIG)

# Debug symbols
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
# Place SWIG shared lib in root directory with python, c++ bindings
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

set(CMAKE_SWIG_FLAGS "")

include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${PYTHON_INCLUDE_PATH})
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

add_library(graph OBJECT graph.cpp)
add_library(DP OBJECT DP.cpp)
add_library(DP_multiprec OBJECT DP_multiprec.cpp)
add_library(LTSS OBJECT LTSS.cpp)

# C++ executable gtest_all
add_executable(gtest_all gtest_all.cpp)
target_link_libraries(DP LTSS)
target_link_libraries(DP_multiprec ${Boost_LIBRARIES} gmp)
target_link_libraries(gtest_all graph LTSS DP ${GTEST_LIBRARIES} pthread)

# C++ executable test_partitions
add_library(test_optimal_partition test_optimal_partition.cpp)
add_executable(test_partitions test_partitions.cpp)
target_link_libraries(test_partitions test_optimal_partition pthread)

# C++ LTSSSolver
add_executable(LTSS_solver_test LTSS_solver_test.cpp)
target_link_libraries(LTSS_solver_test LTSS)

# C++ executable graph_to_dot
add_executable(graph_to_dot graph_to_dot.cpp)
target_link_libraries(graph_to_dot graph)

# C++ executable solver_timer
add_library(timer OBJECT timer.cpp)
add_executable(solver_timer solver_timer.cpp)
target_link_libraries(solver_timer DP LTSS graph timer)

# C++ executable multiprecisision
add_executable(multiprecision multiprecision.cpp)
target_link_libraries(multiprecision ${Boost_LIBRARIES} gmp)

# C++ DPSolver
add_executable(DP_solver_test DP_solver_test.cpp)
target_link_libraries(DP_solver_test DP LTSS graph pthread)

# C++ DP_multiprecSolver
add_executable(DP_multiprec_solver_test DP_multiprec_solver_test.cpp)
target_link_libraries(DP_multiprec_solver_test ${Boost_LIBRARIES} gmp DP_multiprec LTSS DP graph pthread)

# SWIG bindings
set_source_files_properties(proto.i PROPERTIES CPLUSPLUS ON)
# set_source_files_properties(proto.i PROPERTIES SWIG_FLAGS "-includeall")
swig_add_library(test_partition_optimizer LANGUAGE python OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR} OUTFILE_DIR ${CMAKE_CURRENT_SOURCE_DIR} SOURCES python_graph.cpp graph.cpp LTSS.cpp python_dpsolver.cpp DP.cpp python_dp_multisolver.cpp DP_multiprec.cpp python_ltsssolver.cpp LTSS.cpp proto.i)
swig_link_libraries(test_partition_optimizer ${PYTHON_LIBRARIES})

# Equivalent CL swig:
# % swig -c++ -python proto.i
# % g++ -std=c++17 -c -fpic python_graph.cpp proto_wrap.cxx -I/usr/include/python3.6
# % g++ -std=c++17 -shared python_graph.o graph.o proto_wrap.o -o _proto.so -lstdc++