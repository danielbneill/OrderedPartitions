#ifndef __PYTHON_GRAPH_HPP__
#define __PYTHON_GRAPH_HPP__

#include "graph.hpp"
#include "threadpool.hpp"
#include "threadsafequeue.hpp"

#include <vector>
#include <utility>
#include <limits>
#include <type_traits>

std::vector<std::vector<int> > find_optimal_partition__PG(int n,
							  int T,
							  std::vector<float> a,
							  std::vector<float> b);
float find_optimal_weight__PG(int n,
			      int T,
			      std::vector<float> a,
			      std::vector<float> b);

std::pair<std::vector<std::vector<int> >, float> optimize_one__PG(int n,
								  int T,
								  std::vector<float> a,
								  std::vector<float> b);

std::pair<std::vector<std::vector<int> >, float> sweep_best__PG(int n,
								int T,
								std::vector<float> a,
								std::vector<float> b);
std::vector<std::pair<std::vector<std::vector<int>>, float>> sweep_parallel__PG(int n,
										int T,
										std::vector<float> a,
										std::vector<float> b);
std::vector<std::pair<std::vector<std::vector<int>>, float>> sweep__PG(int n,
								       int T,
								       std::vector<float> a,
								       std::vector<float> b);


#endif
