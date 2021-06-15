# OrderedPartitions
Utilities for subset scanning/optimization over partitions of records. There are two sets of executables - optimal partition utilities, and optimizers for classification purposes. The latter are invoked through the Python driver ImpliedBoost.py. The utilities are described below.

# Requirements:
  * For partition testing utilities
    - cmake
    - c++17-compliant compiler
  * For full classifier
    - SWIG
    - theano, for gradient, hessian calculation [only for python-based XGBoost applications]
    - Boost, for BGL for graph solver

To make

$cmake -H. -Bbuild

$cmake --build build -- -j4

binaries:
~/build/bin/

    multiprecision
    test_partitions
    DP_solver_test
    LTSS_solver_test
    solver_timer
    graph_to_dot
    gtest_all
    
## 

# binaries created in ./build/bin

  * test_partitions
    + usage example 
      - // To test (N, T, gamma) = (10, 3, 2.0), e.g.
      - $./build/bin/test_partitions 10 3 2.0

  * gtest_all
    + Google test framework for testing partition utilities, graph-based solver 

## 

# classifier

  * InductiveGBoost.py, classifier based on bottom-up tree specification. Multiple T (partition size) are tested at each iteration step on different threads/tasks, optimal T chosen based on objective function. Work in progress.
