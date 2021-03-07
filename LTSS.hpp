#ifndef __LTSS_HPP__
#define __LTSS_HPP__

#include <list>
#include <utility>
#include <vector>
#include <limits>
#include <iterator>

using ivec = std::vector<int>;
using fvec = std::vector<float>;
using ivecvec = std::vector<std::vector<int>>;
using fvecvec = std::vector<std::vector<float>>;

class LTSSSolver {
public:
  LTSSSolver(int n,
	     fvec a,
	     fvec b
	     ) :
    n_{n},
    a_{a},
    b_{b}
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

  void _init() { create(); optimize(); }
  void create();
  void optimize();

  void sort_by_priority(fvec&, fvec&);
  float compute_score(int, int);
};

#endif
