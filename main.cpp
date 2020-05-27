#include <iostream>
#include <vector>
#include <random>
#include <iterator>

#include "graph.hpp"

auto main(int argc, char **argv) -> int {
  
  int n = 50, T = 5;

  /*
  // Generate random inputs
  std::default_random_engine gen;
  gen.seed(std::random_device()());
  std::uniform_real_distribution<float> dist(1., 10.);
  
  std::vector<float> a(n), b(n);
  
  for (auto &el : a)
    el = dist(gen);
  for (auto &el : b)
    el = dist(gen);
  */

  std::vector<float> a{0.21811402, 0.21811402, 0.21811402, 0.34704381, 0.57778692,
      0.57778692, 0.95346898, 0.95346898, 0.95346898, 0.95346898,
      0.95346898, 0.95346898, 0.95346898, 0.95346898, 0.95346898,
      0.95346898, 0.95346898, 0.95346898, 0.95346898, 0.95346898,
      0.95346898, 0.95346898, 1.04653102, 1.04653102, 1.04653102,
      1.04653102, 1.04653102, 1.04653102, 1.04653102, 1.04653102,
      1.04653102, 1.04653102, 1.04653102, 1.04653102, 1.04653102,
      1.04653102, 1.04653102, 1.04653102, 1.04653102, 1.04653102,
      1.04653102, 1.04653102, 1.42221308, 1.42221308, 1.42221308,
      1.78188598, 1.78188598, 1.78188598, 1.78188598, 1.78188598};
  std::vector<float> b{2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
      2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
      2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.};
  
  auto pg = PartitionGraph(n, T, a, b);

  /*
  // extern c interface
  auto pg_c = PartitionGraph(n, T, a, b);
  std::vector<int> optimalpath_c = pg_c.get_optimal_path_extern();

  int i = 0;
  for (auto &el : optimalpath_c) {
    std::cout << i++ << " : " << el << "\n";
  }
  */

  return 0;
}
