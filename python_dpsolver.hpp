#ifndef __PYTHON_DPSOLVER_HPP__
#define __PYTHON_DPSOLVER_HPP__

#include "score.hpp"
#include "DP.hpp"
#include "threadpool.hpp"
#include "threadsafequeue.hpp"

#include <vector>
#include <utility>
#include <limits>
#include <type_traits>

using ivec = std::vector<int>;
using fvec = std::vector<float>;
using ivecvec = std::vector<ivec>;
using swpair = std::pair<ivecvec, float>;
using swcont = std::vector<swpair>;


ivecvec find_optimal_partition__DP(int n,
				   int T,
				   std::vector<float> a,
				   std::vector<float> b,
				   int parametric_dist,
				   bool risk_partitioning_objective,
				   bool use_rational_optimization
				   );

float find_optimal_score__DP(int n,
			     int T,
			     std::vector<float> a,
			     std::vector<float> b,
			     int parametric_dist,
			     bool risk_partitioning_objective,
			     bool use_rational_optimization);

swpair optimize_one__DP(int n,
			int T,
			std::vector<float> a,
			std::vector<float> b,
			int parametric_dist,
			bool risk_partitioning_objective,
			bool use_rational_optimization);

swpair sweep_best__DP(int n,
		      int T,
		      std::vector<float> a,
		      std::vector<float> b,
		      int parametric_dist,
		      bool risk_partitioning_objective,
		      bool use_rational_optimization);

swcont sweep_parallel__DP(int n,
			  int T,
			  std::vector<float> a,
			  std::vector<float> b,
			  int parametric_dist,
			  bool risk_partitioning_objective,
			  bool use_rational_optimization);

swcont sweep__DP(int n,
		 int T,
		 std::vector<float> a,
		 std::vector<float> b,
		 int parametric_dist,
		 bool risk_partitioning_objective,
		 bool use_rational_optimization);
#endif
