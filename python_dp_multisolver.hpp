#ifndef __PYTHON_DP_MULTISOLVER_HPP__
#define __PYTHON_DP_MULtISOLVER_HPP__

#include "DP_multiprec.hpp"
#include "threadpool.hpp"
#include "threadsafequeue.hpp"

#include <vector>
#include <utility>
#include <limits>
#include <type_traits>

#include <boost/multiprecision/gmp.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

using boost::multiprecision::cpp_dec_float_100;
using namespace boost::multiprecision;

using ivec = std::vector<int>;
using fvec = std::vector<cpp_dec_float_100>;
using ivecvec = std::vector<ivec>;
using swpair = std::pair<ivecvec, cpp_dec_float_100>;
using swcont = std::vector<swpair>;

ivecvec find_optimal_partition__DP_multi(int n,
					 int T,
					 std::vector<cpp_dec_float_100> a,
					 std::vector<cpp_dec_float_100> b);

#endif
