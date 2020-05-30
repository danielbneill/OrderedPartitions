#ifndef __PYTHON_GRAPH_HPP__
#define __PYTHON_GRAPH_HPP__

#include <vector>
#include <utility>

std::vector<std::vector<int>> find_optimal_partition(int n,
						     int T,
						     std::vector<float> a,
						     std::vector<float> b);
float find_optimal_weight(int n,
			  int T,
			  std::vector<float> a,
			  std::vector<float> b);

std::pair<std::vector<std::vector<int>>, float> optimize_one(int n,
							     int T,
							     std::vector<float> a,
							     std::vector<float> b);

std::pair<std::vector<std::vector<int>>, float> sweep(int n,
						      int T,
						      std::vector<float> a,
						      std::vector<float> b);


#endif
