#ifndef __PYTHON_LTSSSOLVER_HPP__
#define __PYTHON_LTSSSOLVER_HPP__

#include "LTSS.hpp"

#include <vector>
#include <type_traits>

using ivec = std::vector<int>;
using fvec = std::vector<float>;

ivec find_optimal_partition__LTSS(int n,
				  std::vector<float> a,
				  std::vector<float> b);
float find_optimal_score__LTSS(int n,
			       std::vector<float> a,
			       std::vector<float> b);

#endif
