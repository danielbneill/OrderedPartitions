#include "python_dp_multisolver.hpp"
#include <thread>

std::vector<std::vector<int>> find_optimal_partition__DP_multi(int n,
					 int T,
					 std::vector<boost::multiprecision::cpp_dec_float_100> a,
					 std::vector<boost::multiprecision::cpp_dec_float_100> b) {
  auto dp = DPSolver_multi(n, T, a, b);
  return dp.get_optimal_subsets_extern();
}

boost::multiprecision::cpp_dec_float_100 find_optimal_score__DP_multi(int n,
					       int T,
					       std::vector<boost::multiprecision::cpp_dec_float_100> a,
					       std::vector<boost::multiprecision::cpp_dec_float_100> b) {
  auto dp = DPSolver_multi(n, T, a, b);
  return dp.get_optimal_score_extern();
}

std::pair<std::vector<std::vector<int>>, boost::multiprecision::cpp_dec_float_100> optimize_one__DP_multi(int n,
													  int T,
													  std::vector<boost::multiprecision::cpp_dec_float_100> a,
													  std::vector<boost::multiprecision::cpp_dec_float_100> b) {
  auto dp = DPSolver_multi(n, T, a, b);
  std::vector<std::vector<int>> subsets = dp.get_optimal_subsets_extern();
  boost::multiprecision::cpp_dec_float_100 score = dp.get_optimal_score_extern();
  
  return std::make_pair(subsets, score);
}
