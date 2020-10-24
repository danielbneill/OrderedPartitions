#include "python_dpsolver.hpp"
#include <thread>

ivecvec find_optimal_partition(int n,
			       int T,
			       std::vector<float> a,
			       std::vector<float> b) {
  auto dp = DPSolver(n, T, a, b);
  return dp.get_optimal_subsets_extern();
}
