#include "python_ltsssolver.hpp"

ivec find_optimal_partition__LTSS(int n,
				  std::vector<float> a,
				  std::vector<float> b) {

  auto ltss = LTSSSolver(n, a, b);
  return ltss.get_optimal_subset_extern();

}

float find_optimal_score__LTSS(int n,
			       std::vector<float> a,
			       std::vector<float> b) {
  auto ltss = LTSSSolver(n, a, b);
  return ltss.get_optimal_score_extern();
}

spair optimize_one__LTSS(int n,
			 std::vector<float> a,
			 std::vector<float> b) {
  auto ltss = LTSSSolver(n, a, b);
  ivec subset = ltss.get_optimal_subset_extern();
  float score = ltss.get_optimal_score_extern();
  return std::make_pair(subset, score);
}
