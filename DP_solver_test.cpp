#include "DP_solver_test.hpp"

void print_subsets(std::vector<std::vector<int>>& subsets) {
  std::cout << "SUBSETS\n";
  std::cout << "[\n";
  std::for_each(subsets.begin(), subsets.end(), [](std::vector<int>& subset){
		  std::cout << "[";
		  std::copy(subset.begin(), subset.end(),
			    std::ostream_iterator<int>(std::cout, " "));
		  std::cout << "]\n";
		});
  std::cout << "]";
}

void sort_by_priority(std::vector<float>& a, std::vector<float>& b) {
  std::vector<int> ind(a.size());
  std::iota(ind.begin(), ind.end(), 0);
  
  std::stable_sort(ind.begin(), ind.end(),
		   [&a, &b](int i, int j) {
		     return (a[i]/b[i]) < (a[j]/b[j]);
		   });
  std::vector<float> a_s, b_s;
  for (auto i : ind) {
    a_s.push_back(a[i]);
    b_s.push_back(b[i]);
  }

  std::copy(a_s.begin(), a_s.end(), a.begin());
  std::copy(b_s.begin(), b_s.end(), b.begin());
	    
}

auto main() -> int {
  
  std::vector<float> a{0.0212651 , -0.20654906, -0.20654906, -0.20654906, -0.20654906,
      0.0212651 , -0.20654906,  0.0212651 , -0.20654906,  0.0212651 ,
      -0.20654906,  0.0212651 , -0.20654906, -0.06581402,  0.0212651 ,
      0.03953075, -0.20654906,  0.16200014,  0.0212651 , -0.20654906,
      0.20296943, -0.18828341, -0.20654906, -0.20654906, -0.06581402,
      -0.20654906,  0.16200014,  0.03953075, -0.20654906, -0.20654906,
      0.03953075,  0.20296943, -0.20654906,  0.0212651 ,  0.20296943,
      -0.20654906,  0.0212651 ,  0.03953075, -0.20654906,  0.03953075};
  std::vector<float> b{0.22771114, 0.21809504, 0.21809504, 0.21809504, 0.21809504,
      0.22771114, 0.21809504, 0.22771114, 0.21809504, 0.22771114,
      0.21809504, 0.22771114, 0.21809504, 0.22682739, 0.22771114,
      0.22745816, 0.21809504, 0.2218354 , 0.22771114, 0.21809504,
      0.218429  , 0.219738  , 0.21809504, 0.21809504, 0.22682739,
      0.21809504, 0.2218354 , 0.22745816, 0.21809504, 0.21809504,
      0.22745816, 0.218429  , 0.21809504, 0.22771114, 0.218429  ,
      0.21809504, 0.22771114, 0.22745816, 0.21809504, 0.22745816};

  // Presort
  sort_by_priority(a, b);
    
  auto dp = DPSolver(40, 5, a, b);
  auto dp_opt = dp.get_optimal_subsets_extern();
  auto scores = dp.get_score_by_subset_extern();

  auto pg = PartitionGraph(40, 5, a, b);
  auto pg_opt = pg.get_optimal_subsets_extern();

  std::cout << "Partition graph subsets:\n";
  std::cout << "=======================\n";
  print_subsets(pg_opt);
  std::cout << "\nDPSolver subsets:\n";
  std::cout << "================\n";
  print_subsets(dp_opt);
  std::cout << "\nScores: ";
  std::copy(scores.begin(), scores.end(), std::ostream_iterator<float>(std::cout, " "));
  std::cout << std::endl;
  
  /*
  std::vector<float> a{1., 2., 3., 4., 5.};
  std::vector<float> b{1., 1., 1., 1., 1.};
  auto dp = DPSolver(5, 3, a, b);
  */

  return 0;
}
