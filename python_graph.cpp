#include <limits>

#include "python_graph.hpp"
#include "graph.hpp"

ivecvec find_optimal_partition(int n, 
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

swpair optimize_one(int n,
		    int T,
		    std::vector<float> a,
		    std::vector<float> b) {

  auto pg = PartitionGraph(n, T, a, b);
  ivecvec subsets = pg.get_optimal_subsets_extern();
  float weight = pg.get_optimal_weight_extern();

  return std::make_pair(subsets, weight);
}

swpair sweep_best(int n,
		  int T,
		  std::vector<float> a,
		  std::vector<float> b) {
  
  float best_weight = std::numeric_limits<float>::max(), weight;
  ivecvec subsets;
  
  for (int i=T; i>1; --i) {
    PartitionGraph pg{n, i, a, b};
    weight = pg.get_optimal_weight_extern();
    std::cout << "NUM_PARTITIONS: " << T << " WEIGHT: " << weight << std::endl;
    if (weight < best_weight) {
      best_weight = weight;
      subsets = pg.get_optimal_subsets_extern();
    }
  }

  return std::make_pair(subsets, weight);
  
}

swcont sweep(int n,
	     int T,
	     std::vector<float> a,
	     std::vector<float> b) {
  

  float weight;
  ivecvec subsets;
  swcont r;

  for (int i=T; i>1; --i) {
    PartitionGraph pg{n, i, a, b};
    weight = pg.get_optimal_weight_extern();
    subsets = pg.get_optimal_subsets_extern();

    r.push_back(std::make_pair(subsets, weight));
  }

  return r;
  
}
