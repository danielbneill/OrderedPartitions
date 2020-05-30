#include <limits>

#include "python_graph.hpp"
#include "graph.hpp"

std::vector<std::vector<int>> find_optimal_partition(int n, 
						     int T, 
						     std::vector<float> a, 
						     std::vector<float> b) {

  auto pg = PartitionGraph(n, T, a, b);
  return pg.get_optimal_subsets_extern();
  
}

float find_optimal_weight(int n,
			  int T,
			  std::vector<float> a,
			  std::vector<float> b) {
  auto pg = PartitionGraph(n, T, a, b);
  return pg.get_optimal_weight_extern();

}

std::pair<std::vector<std::vector<int>>, float> optimize_one(int n,
							     int T,
							     std::vector<float> a,
							     std::vector<float> b) {

  auto pg = PartitionGraph(n, T, a, b);
  std::vector<std::vector<int>> subsets = pg.get_optimal_subsets_extern();
  float weight = pg.get_optimal_weight_extern();

  return std::make_pair(subsets, weight);
}

std::pair<std::vector<std::vector<int>>, float> sweep(int n,
						      int T,
						      std::vector<float> a,
						      std::vector<float> b) {
  
  float best_weight = std::numeric_limits<float>::max(), weight;
  std::vector<std::vector<int>> subsets;

  for (size_t i=T; i>1; --i) {
    PartitionGraph pg{n, T, a, b};
    weight = pg.get_optimal_weight_extern();
    std::cout << "NUM_PARTITIONS: " << T << " WEIGHT: " << weight << std::endl;
    if (weight < best_weight) {
      best_weight = weight;
      subsets = pg.get_optimal_subsets_extern();
    }
  }

  return std::make_pair(subsets, weight);
  
}
