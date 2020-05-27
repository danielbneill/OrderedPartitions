#include "python_graph.hpp"
#include "graph.hpp"

std::vector<int> find_path(int n, int T, std::vector<float> a, std::vector<float> b) {

  auto pg = PartitionGraph(n, T, a, b);
  return pg.get_optimal_path_extern();
  
}
