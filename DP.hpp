#ifndef __DP_HPP__
#define __DP_HPP__

#include <list>
#include <utility>
#include <vector>
#include <limits>
#include <iterator>

using ipair = std::pair<int, int>;
using ipairlist = std::list<ipair>;
using ilist = std::list<int>;
using ivec = std::vector<int>;
using fvec = std::vector<float>;
using ivecvec = std::vector<std::vector<int>>;
using fvecvec = std::vector<std::vector<float>>;

class DPSolver {
public:
  DPSolver(int n,
	   int T,
	   fvec a,
	   fvec b,
	   bool use_rational_optimization=false
	   ) :
    n_{n},
    T_{T},
    a_{a},
    b_{b},
    use_rational_optimization_{use_rational_optimization}
  { _init(); }

  ivecvec get_optimal_subsets_extern() const;
  float get_optimal_score_extern() const;
  void print_maxScore_();
  void print_nextStart_();
    
private:
  int n_;
  int T_;
  fvec a_;
  fvec b_;
  fvecvec a_sums_;
  fvecvec b_sums_;
  fvecvec maxScore_;
  ivecvec nextStart_;
  ivec priority_sortind_;
  float optimal_score_;
  ivecvec subsets_;
  bool use_rational_optimization_;

  void _init() { create(); optimize(); }
  void create();
  void optimize();

  void sort_by_priority(fvec&, fvec&);
  float compute_score_optimized(int, int);
  float compute_score(int, int);
  void compute_partial_sums();

};

#endif
