#ifndef __LTSS_HPP__
#define __LTSS_HPP__

#include <list>
#include <utility>
#include <vector>
#include <limits>
#include <iterator>
#include <algorithm>
#include <memory>

#include "port_utils.hpp"
#include "score.hpp"

using namespace Objectives;

using ivec = std::vector<int>;
using fvec = std::vector<float>;
using ivecvec = std::vector<std::vector<int>>;
using fvecvec = std::vector<std::vector<float>>;

class LTSSSolver {
public:
  LTSSSolver(int n,
	     fvec a,
	     fvec b,
	     objective_fn parametric_dist=objective_fn::Gaussian
	     ) :
    n_{n},
    a_{a},
    b_{b},
    parametric_dist_{parametric_dist}
  { _init(); }

  ivec priority_sortind_;
  ivec get_optimal_subset_extern() const;
  float get_optimal_score_extern() const;

private:
  int n_;
  fvec a_;
  fvec b_;
  float optimal_score_;
  ivec subset_;
  objective_fn parametric_dist_;
  std::unique_ptr<ParametricContext> context_;

  void _init() { create(); optimize(); }
  void create();
  void optimize();

  void sort_by_priority(fvec&, fvec&);
  float compute_score(int, int);
};

#endif
