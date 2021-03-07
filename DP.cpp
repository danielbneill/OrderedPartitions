#include <algorithm>
#include <iterator>
#include <numeric>
#include <limits>
#include <iostream>
#include <utility>
#include <cmath>

#include "DP.hpp"

/*
float
DPSolver::compute_score(int i, int j) {
  float score = std::pow(std::accumulate(a_.begin()+i, a_.begin()+j, 0.), 2) /
    std::accumulate(b_.begin()+i, b_.begin()+j, 0.);
  return score;
}
*/

float
DPSolver::compute_score(int i, int j) {
  float asum = std::accumulate(a_.begin()+i, a_.begin()+j, 0.);
  float bsum = std::accumulate(b_.begin()+i, b_.begin()+j, 0.);
  if (asum > bsum) {
    return asum*std::log(asum/bsum) + bsum - asum;
  } else {
    return 0.;
  }
}

void
DPSolver::sort_by_priority(std::vector<float>& a, std::vector<float>& b) {
  std::vector<int> ind(a.size());
  std::iota(ind.begin(), ind.end(), 0);

  std::stable_sort(ind.begin(), ind.end(),
		   [&a, &b](int i, int j) {
		     return (a[i]/b[i]) < (a[j]/b[j]);
		   });

  priority_sortind_ = ind;

  // Inefficient reordering
  std::vector<float> a_s, b_s;
  for (auto i : ind) {
    a_s.push_back(a[i]);
    b_s.push_back(b[i]);
  }

  std::copy(a_s.begin(), a_s.end(), a.begin());
  std::copy(b_s.begin(), b_s.end(), b.begin());
  
}

void
DPSolver::print_maxScore_() {

  for (int i=0; i<n_; ++i) {
    std::copy(maxScore_[i].begin(), maxScore_[i].end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;
  }
}

void
DPSolver::print_nextStart_() {
  for (int i=0; i<n_; ++i) {
    std::copy(nextStart_[i].begin(), nextStart_[i].end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;
  }
}

void
DPSolver::create() {
  optimal_score_ = 0.;
  
  // sort vectors by priority function G(x,y) = x/y
  sort_by_priority(a_, b_);

  // Initialize matrix
  maxScore_ = std::vector<std::vector<float>>(n_, std::vector<float>(T_+1, std::numeric_limits<float>::min()));
  nextStart_ = std::vector<std::vector<int>>(n_, std::vector<int>(T_+1, -1));
  subsets_ = std::vector<std::vector<int>>(T_, std::vector<int>());

  // Fill in first,second columns corresponding to T = 1,2
  for(int j=0; j<2; ++j) {
    for (int i=0; i<n_; ++i) {
      maxScore_[i][j] = (j==0)?0.:compute_score(i,n_);
      nextStart_[i][j] = (j==0)?-1:n_;
    }
  }

  // Precompute partial sums
  std::vector<std::vector<float>> partialSums;
  partialSums = std::vector<std::vector<float>>(n_, std::vector<float>(n_, 0.));
  for (int i=0; i<n_; ++i) {
    for (int j=i; j<n_; ++j) {
      partialSums[i][j] = compute_score(i, j);
    }
  }

  // Fill in column-by-column from the left
  float score;
  float maxScore;
  int maxNextStart = -1;
  for(int j=2; j<=T_; ++j) {
    for (int i=0; i<n_; ++i) {
      maxScore = std::numeric_limits<float>::min();
      for (int k=i+1; k<=(n_-(j-1)); ++k) {
	score = partialSums[i][k] + maxScore_[k][j-1];
	if (score >= maxScore) {
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
DPSolver::optimize() {
  // Pick out associated maxScores element

  int currentInd = 0, nextInd = 0;
  float score_num = 0., score_den = 0.;
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
DPSolver::get_optimal_subsets_extern() const {
  return subsets_;
}

float
DPSolver::get_optimal_score_extern() const {
  return optimal_score_;
}
