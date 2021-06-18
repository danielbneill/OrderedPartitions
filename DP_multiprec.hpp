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

class DPSolver_multi {
public:
  DPSolver_multi(int n,
	   int T,
	   std::vector<cpp_dec_float_100> a,
	   std::vector<cpp_dec_float_100> b,
	   bool use_rational_optimization=false
	   ) :
    n_{n},
    T_{T},
    a_{a},
    b_{b},
    use_rational_optimization_{use_rational_optimization}
  { _init(); }

  std::vector<std::vector<int>> get_optimal_subsets_extern() const;
  cpp_dec_float_100 get_optimal_score_extern() const;
  void print_maxScore_();
  void print_nextStart_();
    
private:
  int n_;
  int T_;
  std::vector<cpp_dec_float_100> a_;
  std::vector<cpp_dec_float_100> b_;
  std::vector<std::vector<cpp_dec_float_100>> a_sums_;
  std::vector<std::vector<cpp_dec_float_100>> b_sums_;
  std::vector<std::vector<cpp_dec_float_100>> maxScore_;
  std::vector<std::vector<int>> nextStart_;
  std::vector<int> priority_sortind_;
  cpp_dec_float_100 optimal_score_;
  std::vector<std::vector<int>> subsets_;
  bool use_rational_optimization_;

  void _init() { create(); optimize(); }
  void create();
  void optimize();

  void sort_by_priority(std::vector<cpp_dec_float_100>&, std::vector<cpp_dec_float_100>&);
  cpp_dec_float_100 compute_score_optimized(int, int);
  cpp_dec_float_100 compute_score(int, int);
  void compute_partial_sums();

};


#endif
