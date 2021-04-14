#include "python_dp_multisolver.hpp"
#include <thread>

ivecvec find_optimal_partition__DP_multi(int n,
					 int T,
					 std::vector<cpp_dec_float_100> a,
					 std::vector<cpp_dec_float_100> b) {
  auto dp = DPSolver_multi(n, T, a, b);
  return dp.get_optimal_subsets_extern();
}

cpp_dec_float_100 find_optimal_score__DP_multi(int n,
					       int T,
					       std::vector<cpp_dec_float_100> a,
					       std::vector<cpp_dec_float_!00> b) {
  auto dp = DPSolver_multi(n, T, a, b);
  return dp.get_optimal_score_extern();
}
