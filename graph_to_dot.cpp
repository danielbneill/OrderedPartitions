#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iterator>

#include "graph.hpp"

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

auto main(int argc, char **argv) -> int {

  int n, T;

  std::istringstream nss(argv[1]), Tss(argv[2]);
  nss >> n; Tss >> T;

  std::default_random_engine gen;
  gen.seed(std::random_device()());
  std::uniform_real_distribution<float> dista(-10., 10.);
  std::uniform_real_distribution<float> distb(1., 10.);

  std::vector<float> a(n), b(n);

  for (auto &el : a)
    el = dista(gen);
  for (auto &el : b)
    el = distb(gen);

  sort_by_priority(a, b);

  auto pg = PartitionGraph(n, T, a, b);
  auto wt = pg.get_optimal_weight_extern();
  auto subsets = pg.get_optimal_subsets_extern();

  /*
    Details
    std::copy(a.begin(), a.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;
    std::copy(b.begin(), b.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;
    for(size_t i=0; i<a.size(); ++i)
    std::cout << a[i]/b[i] << " ";
    std::cout << std::endl;
    
    std::cout << "SUBSETS\n";
    std::cout << "[\n";
    std::for_each(subsets.begin(), subsets.end(), [](std::vector<int>& subset){
    std::cout << "[";
    std::copy(subset.begin(), subset.end(),
    std::ostream_iterator<int>(std::cout, " "));
    std::cout << "]\n";
    });
    std::cout << "]";
  */
    
  pg.write_dot();

  return 0;
}
