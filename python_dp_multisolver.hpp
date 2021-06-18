#ifndef __PYTHON_DP_MULTISOLVER_HPP__
#define __PYTHON_DP_MULTISOLVER_HPP__

#include "DP_multiprec.hpp"
#include "threadpool.hpp"
#include "threadsafequeue.hpp"

#include <vector>
#include <utility>
#include <limits>
#include <type_traits>

#include <boost/multiprecision/gmp.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

#include "score.hpp"

// using boost::multiprecision::cpp_dec_float_100;
// using namespace boost::multiprecision;

std::vector<std::vector<int>> find_optimal_partition__DP_multi(int n,
							       int T,
							       std::vector<boost::multiprecision::cpp_dec_float_100> a,
							       std::vector<boost::multiprecision::cpp_dec_float_100> b);

boost::multiprecision::cpp_dec_float_100 find_optimal_score__DP_multi(int n,
					       int T,
					       std::vector<boost::multiprecision::cpp_dec_float_100> a,
					       std::vector<boost::multiprecision::cpp_dec_float_100> b);

std::pair<std::vector<std::vector<int>>, boost::multiprecision::cpp_dec_float_100> optimize_one__DP_multi(int n,
													  int T,
													  std::vector<boost::multiprecision::cpp_dec_float_100> a,
													  std::vector<boost::multiprecision::cpp_dec_float_100> b);

#endif
