#ifndef __PYTHON_GRAPH_HPP__
#define __PYTHON_GRAPH_HPP__

#include <vector>
#include <utility>

using ivecvec = std::vector<std::vector<int>>;
using swpair = std::pair<ivecvec, float>;
using swcont = std::vector<swpair>;


ivecvec find_optimal_partition(int n,
			       int T,
			       std::vector<float> a,
			       std::vector<float> b);
float find_optimal_weight(int n,
			  int T,
			  std::vector<float> a,
			  std::vector<float> b);

swpair optimize_one(int n,
		   int T,
		   std::vector<float> a,
		   std::vector<float> b);

swpair sweep_best(int n,
		  int T,
		  std::vector<float> a,
		  std::vector<float> b);
swcont sweep(int n,
	     int T,
	     std::vector<float> a,
	     std::vector<float> b);


#endif
