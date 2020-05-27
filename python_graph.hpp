#ifndef __PYTHON_GRAPH_HPP__
#define __PYTHON_GRAPH_HPP__

#include <vector>

std::vector<std::vector<int>> find_optimal_partition(int n,
						     int T,
						     std::vector<float> a,
						     std::vector<float> b);

#endif
