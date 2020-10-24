#ifndef __PYTHON_DPSOLVER_HPP__
#define __PYTHON_DPSOLVER_HPP__

#include "DP.hpp"
#include "threadpool.hpp"
#include "threadsafequeue.hpp"

#include <vector>
#include <utility>
#include <limits>
#include <type_traits>

using ivec = std::vector<int>;
using fvec = std::vector<float>;
using ivecvec = std::vector<ivec>;
using swpair = std::pair<ivectvec, float>;
using swcont = std::vector<swpair>;


ivecvec find_optimal_partition(int n,
			       int T,
			       std::vector<float> a,
			       std::vector<float> b);

#endif
