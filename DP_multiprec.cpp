#include <algorithm>
#include <iterator>
#include <numeric>
#include <limits>
#include <iostream>
#include <utility>
#include <cmath>

#include "DP_multiprec.hpp"

cpp_dec_float_100
DPSolver_multi::compute_score_optimized(int i, int j) {
  cpp_dec_float_100 score = a_sums_[i][j] / b_sums_[i][j];
  return score;
}

cpp_dec_float_100
DPSolver_multi::compute_score(int i, int j) {
  cpp_dec_float_100 num_ = std::accumulate(a_.begin()+i, a_.begin()+j, static_cast<cpp_dec_float_100>(0.));
  cpp_dec_float_100 den_ = std::accumulate(b_.begin()+i, b_.begin()+j, static_cast<cpp_dec_float_100>(0.));
  cpp_dec_float_100 score = num_*num_/den_;
  return score;
}

void
DPSolver_multi::compute_partial_sums() {
  cpp_dec_float_100 a_cum;
  a_sums_ = std::vector<std::vector<cpp_dec_float_100>>(n_, std::vector<cpp_dec_float_100>(n_+1, 
											   static_cast<cpp_dec_float_100>(std::numeric_limits<float>::min())));
  b_sums_ = std::vector<std::vector<cpp_dec_float_100>>(n_, std::vector<cpp_dec_float_100>(n_+1, 
											   static_cast<cpp_dec_float_100>(std::numeric_limits<float>::min())));

  for (int i=0; i<n_; ++i) {
    a_sums_[i][i] = 0.;
    b_sums_[i][i] = 0.;
  }

  for (int i=0; i<n_; ++i) {
    a_cum = (i == 0)? static_cast<cpp_dec_float_100>(0.) : -a_[i-1];
    for (int j=i+1; j<=n_; ++j) {
      a_cum += (j >= 2)? a_[j-2] : static_cast<cpp_dec_float_100>(0.);
      a_sums_[i][j] = a_sums_[i][j-1] + (2*a_cum + a_[j-1])*a_[j-1];
      b_sums_[i][j] = b_sums_[i][j-1] + b_[j-1];
    }
  }      
}

void
DPSolver_multi::sort_by_priority(std::vector<cpp_dec_float_100>& a, std::vector<cpp_dec_float_100>& b) {
  std::vector<int> ind(a.size());
  std::iota(ind.begin(), ind.end(), 0);

  std::stable_sort(ind.begin(), ind.end(),
		   [&a, &b](int i, int j) {
		     return (a[i]/b[i]) < (a[j]/b[j]);
		   });

  priority_sortind_ = ind;

  // Inefficient reordering
  std::vector<cpp_dec_float_100> a_s, b_s;
  for (auto i : ind) {
    a_s.push_back(a[i]);
    b_s.push_back(b[i]);
  }

  std::copy(a_s.begin(), a_s.end(), a.begin());
  std::copy(b_s.begin(), b_s.end(), b.begin());
  
}

void
DPSolver_multi::print_maxScore_() {

  for (int i=0; i<n_; ++i) {
    std::copy(maxScore_[i].begin(), maxScore_[i].end(), std::ostream_iterator<cpp_dec_float_100>(std::cout, " "));
    std::cout << std::endl;
  }
}

void
DPSolver_multi::print_nextStart_() {
  for (int i=0; i<n_; ++i) {
    std::copy(nextStart_[i].begin(), nextStart_[i].end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
  }
}

void
DPSolver_multi::create() {
  optimal_score_ = 0.;
  cpp_dec_float_100 (DPSolver_multi::*score_function)(int,int) = &DPSolver_multi::compute_score;
  
  // sort vectors by priority function G(x,y) = x/y
  sort_by_priority(a_, b_);

  // Initialize matrix
  maxScore_ = std::vector<std::vector<cpp_dec_float_100>>(n_, std::vector<cpp_dec_float_100>(T_+1, 
											     static_cast<cpp_dec_float_100>(std::numeric_limits<float>::min())));
  nextStart_ = std::vector<std::vector<int>>(n_, std::vector<int>(T_+1, -1));
  subsets_ = std::vector<std::vector<int>>(T_, std::vector<int>());

  // Precompute partial sums
  if (use_rational_optimization_) {
    compute_partial_sums();
    score_function = &DPSolver_multi::compute_score_optimized;
  }

  // Fill in first,second columns corresponding to T = 1,2
  for(int j=0; j<2; ++j) {
    for (int i=0; i<n_; ++i) {
      maxScore_[i][j] = (j==0)?0.:(this->*score_function)(i,n_);
      nextStart_[i][j] = (j==0)?-1:n_;
    }
  }

  // Precompute partial sums
  std::vector<std::vector<cpp_dec_float_100>> partialSums;
  partialSums = std::vector<std::vector<cpp_dec_float_100>>(n_, std::vector<cpp_dec_float_100>(n_, 0.));
  for (int i=0; i<n_; ++i) {
    for (int j=i; j<n_; ++j) {
      partialSums[i][j] = (this->*score_function)(i, j);
    }
  }

  // Fill in column-by-column from the left
  cpp_dec_float_100 score;
  cpp_dec_float_100 maxScore;
  int maxNextStart = -1;
  for(int j=2; j<=T_; ++j) {
    for (int i=0; i<n_; ++i) {
      maxScore = static_cast<cpp_dec_float_100>(std::numeric_limits<float>::min());
      for (int k=i+1; k<=(n_-(j-1)); ++k) {
	score = partialSums[i][k] + maxScore_[k][j-1];
	if (score > maxScore) {
	  maxScore = score;
	  maxNextStart = k;
	}
      }
      maxScore_[i][j] = maxScore;
      nextStart_[i][j] = maxNextStart;
      // Only need the initial entry in last column
      if (j == T_)
	break;
    }
  }
}

void
DPSolver_multi::optimize() {
  // Pick out associated maxScores element
  int currentInd = 0, nextInd = 0;
  cpp_dec_float_100 score_num = 0., score_den = 0.;
  for (int t=T_; t>0; --t) {
    nextInd = nextStart_[currentInd][t];
    for (int i=currentInd; i<nextInd; ++i) {
      subsets_[T_-t].push_back(priority_sortind_[i]);
      score_num += a_[priority_sortind_[i]];
      score_den += b_[priority_sortind_[i]];
    }
    optimal_score_ += score_num*score_num/score_den;
    currentInd = nextInd;
  }
}

ivecvec
DPSolver_multi::get_optimal_subsets_extern() const {
  return subsets_;
}

cpp_dec_float_100
DPSolver_multi::get_optimal_score_extern() const {
  return optimal_score_;
}
