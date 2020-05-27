#include "python_graph.hpp"
#include "graph.hpp"

std::vector<std::vector<int>> find_optimal_partition(int n, 
						     int T, 
						     std::vector<float> a, 
						     std::vector<float> b) {

  auto pg = PartitionGraph(n, T, a, b);
  return pg.get_optimal_subsets_extern();
  
}
