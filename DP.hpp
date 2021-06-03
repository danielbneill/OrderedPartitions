#ifndef __DP_HPP__
#define __DP_HPP__

#include <list>
#include <utility>
#include <vector>
#include <iostream>
#include <limits>
#include <iterator>
#include <memory>
#include <algorithm>
#include <numeric>
#include <cmath>

using ipair = std::pair<int, int>;
using ipairlist = std::list<ipair>;
using ilist = std::list<int>;
using ivec = std::vector<int>;
using fvec = std::vector<float>;
using ivecvec = std::vector<std::vector<int>>;
using fvecvec = std::vector<std::vector<float>>;

enum class objective_fn { Gaussian, Poisson };

class ParametricContext {
protected:
  fvec a_;
  fvec b_;
  int n_;
  fvecvec a_sums_;
  fvecvec b_sums_;
  objective_fn parametric_dist_;
  bool risk_partitioning_objective_;
  bool use_rational_optimization_;

public:
  ParametricContext(fvec a, 
		    fvec b, 
		    int n, 
		    objective_fn parametric_dist,
		    bool risk_partitioning_objective,
		    bool use_rational_optimization
		    ) :
    a_{a},
    b_{b},
    n_{n},
    parametric_dist_{parametric_dist},
    risk_partitioning_objective_{risk_partitioning_objective},
    use_rational_optimization_{use_rational_optimization}

  {}

  virtual ~ParametricContext() = default;

  virtual void compute_partial_sums() { std::cout << "SHOULD NOT REACH THIS" << std::endl; };

  virtual float compute_score_exp(int, int) = 0;
  virtual float compute_score_exp_optimized(int, int) {};
  virtual float compute_score_pop(int, int) = 0;
  virtual float compute_score_pop_optimized(int, int) {};

  float compute_score(int i, int j) {
    if (risk_partitioning_objective_) {
      if (use_rational_optimization_) {
	return compute_score_pop_optimized(i, j);
      }
      else {
	return compute_score_pop(i, j);
      }
    }
    else {
      if (use_rational_optimization_) {
	return compute_score_exp_optimized(i, j);
      }
      else {
	return compute_score_exp(i, j);
      }
    }
  }
};
  
class PoissonContext : public ParametricContext {

public:
  PoissonContext(fvec a, 
		 fvec b, 
		 int n, 
		 objective_fn parametric_dist,
		 bool risk_partitioning_objective,
		 bool use_rational_optimization) : ParametricContext(a,
								     b,
								     n,
								     parametric_dist,
								     risk_partitioning_objective,
								     use_rational_optimization)
  {}

  float compute_score_exp(int i, int j) override {    
    float Cin = std::accumulate(a_.begin()+i, a_.begin()+j, 0.);
    float Bin = std::accumulate(b_.begin()+i, b_.begin()+j, 0.);
    float Cout = std::accumulate(a_.begin(), a_.end(), 0.) - Cin;
    float Bout = std::accumulate(b_.begin(), b_.end(), 0.) - Bin;
    if ((Cin/Bin)>(Cout/Bout)) {
      return Cin*std::log(Cin/Bin) + Cout*std::log(Cout/Bout);
    } else {
      return 0.;
    }
  }

  float compute_score_pop(int i, int j) override {
    return 0.;
  }
  
};

class GaussianContext : public ParametricContext {

public:
  GaussianContext(fvec a, 
		  fvec b, 
		  int n, 
		  objective_fn parametric_dist,
		  bool risk_partitioning_objective,
		  bool use_rational_optimization) : ParametricContext(a,
								      b,
								      n,
								      parametric_dist,
								      risk_partitioning_objective,
								      use_rational_optimization)
  { if (use_rational_optimization) {
      compute_partial_sums();
    }
  }
  
  void compute_partial_sums() override {
    float a_cum;
    a_sums_ = std::vector<std::vector<float>>(n_, std::vector<float>(n_+1, std::numeric_limits<float>::min()));
    b_sums_ = std::vector<std::vector<float>>(n_, std::vector<float>(n_+1, std::numeric_limits<float>::min()));
    
    for (int i=0; i<n_; ++i) {
      a_sums_[i][i] = 0.;
      b_sums_[i][i] = 0.;
    }

    for (int i=0; i<n_; ++i) {
      a_cum = -a_[i-1];
      for (int j=i+1; j<=n_; ++j) {
	a_cum += a_[j-2];
	a_sums_[i][j] = a_sums_[i][j-1] + (2*a_cum + a_[j-1])*a_[j-1];
	b_sums_[i][j] = b_sums_[i][j-1] + b_[j-1];
      }
    }  
  }
  
  float compute_score_exp_optimized(int i, int j) override {
    float score = a_sums_[i][j] / b_sums_[i][j];
    return score;
  }

  float compute_score_exp(int i, int j) override {
    float score = std::pow(std::accumulate(a_.begin()+i, a_.begin()+j, 0.), 2) /
      std::accumulate(b_.begin()+i, b_.begin()+j, 0.);
    return score;
  }
  
  float compute_score_pop(int i, int j) override {
    float Cin = std::accumulate(a_.begin()+i, a_.begin()+j, 0.);
    float Bin = std::accumulate(b_.begin()+i, b_.begin()+j, 0.);
    float Cout = std::accumulate(a_.begin(), a_.end(), 0.) - Cin;
    float Bout = std::accumulate(b_.begin(), b_.end(), 0.) - Bin;
    if ((Cin/Bin)>(Cout/Bout)) {
      return (Cin*Cin/2./Bin) + (Cout*Cout/2./Bout);
    } else {
      return 0.;
    }
  }

};

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
  fvecvec maxScore_;
  ivecvec nextStart_;
  ivec priority_sortind_;
  float optimal_score_;
  ivecvec subsets_;
  fvec score_by_subset_;
  objective_fn parametric_dist_;
  bool risk_partitioning_objective_;
  bool use_rational_optimization_;
  std::unique_ptr<ParametricContext> context_;

  void _init() { create(); optimize(); }
  void create();
  void optimize();

  void sort_by_priority(fvec&, fvec&);
  float compute_score(int, int);
};


#endif
