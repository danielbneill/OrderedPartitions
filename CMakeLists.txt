cmake_minimum_required(VERSION 3.6)

project(proto)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)

find_package(SWIG REQUIRED)
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)

include(UseSWIG)
# set(UseSWIG TARGET_NAME_PREFERENCE STANDARD)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
# Place SWIG shared lib in root directory with python, c++ bindings
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

set(CMAKE_SWIG_FLAGS "")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${PYTHON_INCLUDE_PATH})

add_library(graph OBJECT graph.cpp)

# C++ executable
add_executable(proto main.cpp)
target_link_libraries(proto graph)

# SWIG bindings
set_source_files_properties(proto.i PROPERTIES CPLUSPLUS ON)
# set_source_files_properties(proto.i PROPERTIES SWIG_FLAGS "-includeall")
swig_add_library(proto LANGUAGE python OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR} OUTFILE_DIR ${CMAKE_CURRENT_SOURCE_DIR} SOURCES python_graph.cpp graph.cpp proto.i)
swig_link_libraries(proto ${PYTHON_LIBRARIES})

# Equivalent swig:
# % swig -c++ -python proto.i
# % g++ -std=c++17 -c -fpic python_graph.cpp proto_wrap.cxx -I/usr/include/python3.6
# % g++ -std=c++17 -shared python_graph.o graph.o proto_wrap.o -o _proto.so -lstdc++