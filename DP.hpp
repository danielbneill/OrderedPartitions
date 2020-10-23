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
	   fvec b
	   ) :
    n_{n},
    T_{T},
    a_{a},
    b_{b}
  { _init(); }

  ivecvec get_optimal_subsets_extern() const;
    
private:
  int n_;
  int T_;
  fvec a_;
  fvec b_;
  fvecvec maxScore_;
  ivecvec nextStart_;
  ivec priority_sortind_;
  ivecvec subsets_;

  void _init() { create(); optimize(); }
  void create();
  void optimize();

  void sort_by_priority(fvec&, fvec&);
  float compute_score(int, int);
  void print_maxScore_();
  void print_nextStart_();

};

#endif
