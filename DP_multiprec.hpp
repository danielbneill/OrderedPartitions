#ifndef __DPP_MULTIPREC_HPP__
#define __DPP_MULTIPREC_HPP__

#include <list>
#include <utility>
#include <vector>
#include <limits>
#include <iterator>

#include <boost/multiprecision/gmp.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

using boost::multiprecision::cpp_dec_float_100;
using namespace boost::multiprecision;

using ipair = std::pair<int, int>;
using ipairlist = std::list<ipair>;
using ilist = std::list<int>;
using ivec = std::vector<int>;
using gmpfvec = std::vector<cpp_dec_float_100>;
using ivecvec = std::vector<std::vector<int>>;
using gmpfvecvec = std::vector<std::vector<cpp_dec_float_100>>;

class DPSolver_multi {
public:
  DPSolver_multi(int n,
	   int T,
	   gmpfvec a,
	   gmpfvec b
	   ) :
    n_{n},
    T_{T},
    a_{a},
    b_{b}
  { _init(); }

  ivecvec get_optimal_subsets_extern() const;
  cpp_dec_float_100 get_optimal_score_extern() const;
  void print_maxScore_();
  void print_nextStart_();
    
private:
  int n_;
  int T_;
  gmpfvec a_;
  gmpfvec b_;
  gmpfvecvec a_sums_;
  gmpfvecvec b_sums_;
  gmpfvecvec maxScore_;
  ivecvec nextStart_;
  ivec priority_sortind_;
  cpp_dec_float_100 optimal_score_;
  ivecvec subsets_;

  void _init() { create(); optimize(); }
  void create();
  void optimize();

  void sort_by_priority(gmpfvec&, gmpfvec&);
  cpp_dec_float_100 compute_score(int, int);
  void compute_partial_sums();

};


#endif
