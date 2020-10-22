#ifndef __DP_HPP__
#define __DP_HPP__

#include <list>
#include <utility>
#include <vector>

using ipair = std::pair<int, int>;
using ipairlist = std::list<ipair>;
using ilist = std::list<int>;
using ivec = std::vector<int>;
using fvec = std::vector<float>;
using ivecvec = std::vector<std::vector<int>>;

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
    
private:
  int n_;
  int T_;
  fvec a_;
  fvec b_;

  void _init() { create(); optimize(); }
  void create();
  void optimize();

  ivecvec = get_optimal_subsets_extern() const;

};

#endif
