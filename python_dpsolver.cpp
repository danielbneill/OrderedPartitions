#include "python_dpsolver.hpp"
#include <thread>

using namespace Objectives;

#define MULT_CLUST 1
// #undef MULT_CLUST

#define DPSOLVER_RISK_PART_(n,T,a,b)  (DPSolver(n, T, a, b, objective_fn::Gaussian, true, false))
#define DPSOLVER_MULT_CLUST_(n,T,a,b) (DPSolver(n, T, a, b, objective_fn::Gaussian, false, true))

#ifdef MULT_CLUST
#define DPSOLVER_(n,T,a,b) (DPSOLVER_MULT_CLUST_(n,T,a,b))
#else
#define DPSOLVER_(n,T,a,b) (DPSOLVER_RISK_PART_(n,T,a,b))
#endif

ivecvec find_optimal_partition__DP(int n,
			       int T,
			       std::vector<float> a,
			       std::vector<float> b) {
  auto dp = DPSOLVER_(n, T, a, b);
  return dp.get_optimal_subsets_extern();
}

float find_optimal_score__DP(int n,
			     int T,
			     std::vector<float> a,
			     std::vector<float> b) {
  auto dp = DPSOLVER_(n, T, a, b);
  return dp.get_optimal_score_extern();
}

swpair optimize_one__DP(int n,
			int T,
			std::vector<float> a,
			std::vector<float> b) {
  auto dp = DPSOLVER_(n, T, a, b);
  ivecvec subsets = dp.get_optimal_subsets_extern();
  float score = dp.get_optimal_score_extern();
  
  return std::make_pair(subsets, score);
}

swpair sweep_best__DP(int n,
		      int T,
		      std::vector<float> a,
		      std::vector<float> b) {
  float best_score = std::numeric_limits<float>::max(), score;
  ivecvec subsets;

  for (int i=T; i>1; --i) {
    auto dp = DPSOLVER_(n, i, a, b);
    // XXX
    // Taking minimum here?
    score = dp.get_optimal_score_extern();
    std::cout << "NUM_PARTITIONS: " << T << " SCORE: " << score << std::endl;
    if (score < best_score) {
      best_score = score;
      subsets = dp.get_optimal_subsets_extern();
    }
  }

  return std::make_pair(subsets, score);
}

swcont sweep_parallel__DP(int n,
			  int T,
			  std::vector<float> a,
			  std::vector<float> b) {

  ThreadsafeQueue<swpair> results_queue;

  auto task = [&results_queue](int n, int i, fvec a, fvec b) {
    auto dp = DPSOLVER_(n, i, a, b);
    results_queue.push(std::make_pair(dp.get_optimal_subsets_extern(),
				      dp.get_optimal_score_extern()));
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

swcont sweep__DP(int n,
		 int T,
		 std::vector<float> a,
		 std::vector<float> b) {
  float score;
  ivecvec subsets;
  swcont r;

  for (int i=T; i>1; --i) {
    auto dp = DPSOLVER_(n, i, a, b);
    score = dp.get_optimal_score_extern();
    subsets = dp.get_optimal_subsets_extern();

    r.push_back(std::make_pair(subsets, score));
  }

  return r;
  
}
