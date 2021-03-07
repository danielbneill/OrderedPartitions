#include <algorithm>
#include <iterator>
#include <numeric>
#include <limits>
#include <iostream>
#include <utility>
#include <cmath>

#include "LTSS.hpp"


float
LTSSSolver::compute_score(int i, int j) {
  float score = std::pow(std::accumulate(a_.begin()+i, a_.begin()+j, 0.), 2) /
    std::accumulate(b_.begin()+i, b_.begin()+j, 0.);
  return score;
}

/*
float
LTSSSolver::compute_score(int i, int j) {
  float asum = std::accumulate(a_.begin()+i, a_.begin()+j, 0.);
  float bsum = std::accumulate(b_.begin()+i, b_.begin()+j, 0.);
  if (asum > bsum) {
    return asum*std::log(asum/bsum) + bsum - asum;
  } else {
    return 0.;
  }
}
*/

void
LTSSSolver::sort_by_priority(std::vector<float>& a, std::vector<float>& b) {
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
LTSSSolver::create() {
  // sort by priority
  sort_by_priority(a_, b_);

  subset_ = std::vector<int>();
}

void
LTSSSolver::optimize() {
  optimal_score_ = 0.;

  float maxScore = std::numeric_limits<float>::min();
  std::pair<int, int> p;
  // Test ascending partitions
  for (int i=0; i<n_; ++i) {
    auto score = compute_score(0, i);
    // std::cout << "asc i: " << i << " score: " << score << "\n";
    if (score > maxScore) {
      maxScore = score;
      p = std::make_pair(0, i);
    }
  }
  // Test descending partitions
  for (int i=n_; i>0; --i) {
    auto score = compute_score(i, n_);
    // std::cout << "dsc i: " << i << " score: " << score << "\n";
    if (score > maxScore) {
      maxScore = score;
      p = std::make_pair(i, n_);
    }
  }
  
  for (int i=p.first; i<p.second; ++i) {
    subset_.push_back(i);
  }
  optimal_score_ = maxScore;
}

ivec
LTSSSolver::get_optimal_subset_extern() const {
  return subset_;
}

float
LTSSSolver::get_optimal_score_extern() const {
  return optimal_score_;
}
