#ifndef __DP_HPP__
#define __DP_HPP__

#include <list>
#include <utility>
#include <vector>
#include <iostream>
#include <iterator>
#include <memory>
#include <algorithm>
#include <numeric>
#include <cmath>

#include "score.hpp"
#include "LTSS.hpp"

#define UNUSED(expr) do { (void)(expr); } while (0)

using ipair = std::pair<int, int>;
using ipairlist = std::list<ipair>;
using ilist = std::list<int>;
using ivec = std::vector<int>;
using fvec = std::vector<float>;
using ivecvec = std::vector<std::vector<int>>;
using fvecvec = std::vector<std::vector<float>>;

using namespace Objectives;

class DPSolver {
public:
  DPSolver(int n,
	   int T,
	   fvec a,
	   fvec b,
	   objective_fn parametric_dist=objective_fn::Gaussian,
	   bool risk_partitioning_objective=false,
	   bool use_rational_optimization=false
	   ) :
    n_{n},
    T_{T},
    a_{a},
    b_{b},
    optimal_score_{0.},
    parametric_dist_{parametric_dist},
    risk_partitioning_objective_{risk_partitioning_objective},
    use_rational_optimization_{use_rational_optimization}
    
  { _init(); }

  ivecvec get_optimal_subsets_extern() const;
  float get_optimal_score_extern() const;
  fvec get_score_by_subset_extern() const;
  void print_maxScore_();
  void print_nextStart_();
    
private:
  int n_;
  int T_;
  fvec a_;
  fvec b_;
  fvecvec maxScore_, maxScore_sec_;
  ivecvec nextStart_, nextStart_sec_;
  ivec priority_sortind_;
  float optimal_score_;
  ivecvec subsets_;
  fvec score_by_subset_;
  objective_fn parametric_dist_;
  bool risk_partitioning_objective_;
  bool use_rational_optimization_;
  std::unique_ptr<ParametricContext> context_;
  std::unique_ptr<LTSSSolver> LTSSSolver_;

  void _init() { 
    if (risk_partitioning_objective_) {
      create();
      optimize();
    }
    else {
      create_multiple_clustering_case();
      optimize_multiple_clustering_case();
    }
  }
  void create();
  void create_multiple_clustering_case();
  void optimize();
  void optimize_multiple_clustering_case();

  void sort_by_priority(fvec&, fvec&);
  void reorder_subsets(ivecvec&, fvec&);
  float compute_score(int, int);
  float compute_ambient_score(float, float);
};


#endif
