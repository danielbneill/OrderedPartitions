# OrderedPartitions
Utilities for subset scanning/optimization over partitions of records. There are two sets of executables - optimal partition utilities, and optimizers for classification purposes. The latter are invoked through the Python driver Implied_Boost.py. The utilities are described below.

Requirements:
  * For partition testing utilities
    - cmake
    - c++17-compliant compiler
  * For full classifier
    - SWIG
    - theano, only for full SWIG-based classifier code, not needed for partition testing
    - boost, needed for BGL for graph solver

To make
$cmake -H. -Bbuild
$cmake --build build -- -j4

binaries created in ./build/bin

## 
  * test_partitions
    + usage example 
      // To test (N, T, gamma) = (10, 3, 2.0) case
      $./build/bin/test_partitions 10 3 2.0

  * gtest_all
    + Google test framework for testing partition utilities, graph-based solver 