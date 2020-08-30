#include "python_graph.hpp"
#include <thread>

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

swcont sweep_parallel(int n,
		      int T,
		      std::vector<float> a,
		      std::vector<float> b) {
  
  ThreadsafeQueue<swpair> results_queue;
  
  auto task = [&results_queue](int n, int i, fvec a, fvec b){
    PartitionGraph pg{n, i, a, b};
    results_queue.push(std::make_pair(pg.get_optimal_subsets_extern(),
				      pg.get_optimal_weight_extern()));
  };

  std::vector<ThreadPool::TaskFuture<void>> v;
  
  for (int i=T; i>1; --i) {
    v.push_back(DefaultThreadPool::submitJob(task, n, i, a, b));
  }	       
  for (auto& item : v) 
    item.get();

  swpair result;
  swcont results;
  while (!results_queue.empty()) {
    bool valid = results_queue.waitPop(result);
    if (valid) {
      results.push_back(result);
    }
  }

  return results;

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
