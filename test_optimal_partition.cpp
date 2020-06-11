#include "test_optimal_partition.hpp"

#include <iomanip>

namespace combinatorics {
  // Number of partitions of n of size k
  unsigned long Bell_n_k(int n, int k) {
    if (n == 0 or k == 0 or k > n)
      return 0;
    if (k == 1 or k == n)
      return 1;
    
    return (k * Bell_n_k(n-1, k) + Bell_n_k(n-1, k-1));
  }

  unsigned long nChoosek( unsigned long n, unsigned long k )
  {
    if (k > n) return 0;
    if (k * 2 > n) k = n-k;
    if (k == 0) return 1;
    
    int result = n;
    for( int i = 2; i <= k; ++i ) {
      result *= (n-i+1);
      result /= i;
    }
    return result;
  }

  unsigned long Mon_n_k(int n, int k) {
    // Number of monotonic partitions of n of size k
    return nChoosek(n-1, k-1);
  }

  double mon_to_all_ratio(int n, int k) {
    return static_cast<float>(Mon_n_k(n, k))/static_cast<float>(Bell_n_k(n, k));
  }

} // namespace


void
PartitionTest::sort_by_priority_(std::vector<double>& a, std::vector<double>& b) {
  // TODO: Make this in-place, don't see a good way though

  std::vector<int> ind(a_.size());
  std::iota(ind.begin(), ind.end(), 0);
  stable_sort(ind.begin(), ind.end(),
	      [&a, &b](int i, int j) {
		return (a[i]/b[i]) < (a[j]/b[j]);
	      });
  std::vector<double> a_s(a.size()), b_s(a.size());
  for (size_t i=0; i<a.size(); ++i) {
    a_s[i] = a[ind[i]];
    b_s[i] = b[ind[i]];
  }
  
  std::copy(a_s.begin(), a_s.end(), a.begin());
  std::copy(b_s.begin(), b_s.end(), b.begin());  
}

void
PartitionTest::runTest() {
  int NUM_TASKS = 10;

  int numPartitions = fList_.size();
  int window = static_cast<int>(numPartitions/NUM_TASKS);
  int excess = numPartitions - window*NUM_TASKS;

  sort_by_priority_(a_, b_);

  auto task = [this](int b, int e) {
    results_queue_.push(optimize_(b, e));
  };

  // submit work
  if (window > 10000) {
    for (int i=0; i<NUM_TASKS; ++i) {
      v_.push_back(DefaultThreadPool::submitJob(task, i*window, (i+1)*window));
    }
    // submit remainder of work, if any
    if (excess) { v_.push_back(DefaultThreadPool::submitJob(task, NUM_TASKS*window, numPartitions)); }
    
  } else {
    v_.push_back(DefaultThreadPool::submitJob(task, 0, numPartitions));
  }

  for (auto& item : v_)
    item.get();

  resultPair result;
  while (!results_queue_.empty()) {
    bool valid = results_queue_.waitPop(result);
    if (valid) {
      results_.push_back(result);
    }
  }
  optimalResult_ = results_[0];
}

void
PartitionTest::print_partitions() const {
  for (auto& list : fList_) {
    for (auto& el: list) {
      std::cout << "[ ";
      for (auto& pt: el) {
	std::cout << pt << " ";
      }
      std::cout << " ]";
    }
    std::cout << "\n";    
  }
}

void
PartitionTest::print_pair(const resultPair& p) const {
  std::cout << std::setprecision(12) << "ratio: " << p.first << " ";
  for (auto& el: p.second) {
    std::cout << "[ ";
    for (auto& pt: el) {
      std::cout << pt << " ";
    }
    std::cout << " ]";
  }
  std::cout << "\n";
}

void
PartitionTest::init_(bool formPartitions) {
  elements_ = std::vector<int>(numElements_);
  std::iota(elements_.begin(), elements_.end(), 0);
  if (formPartitions) {
    formPartitions_();
  }
}

void
PartitionTest::cleanup_() {
  results_queue_.clear();
  results_.clear();
}

resultPair
PartitionTest::optimize_(int b, int e) {
  double rSum, paSum, pbSum;
  double rMax = std::numeric_limits<double>::min();
  std::vector<std::vector<int>> partMax;

  for (auto it=fList_.cbegin(); it!=fList_.cend(); ++it) {
    rSum = 0.;
    for (auto pit=(*it).cbegin(); pit!=(*it).cend(); ++pit) {
      paSum = 0.;
      pbSum = 0.;
      for (auto eit=(*pit).cbegin(); eit!=(*pit).cend(); ++eit) {
	paSum += a_[*eit];
	pbSum += b_[*eit];
      }
      rSum += std::pow(paSum, gamma_)/pbSum;
    }
    // print_pair(std::make_pair(rSum, *it));
    if (rSum > rMax) {
      rMax = rSum;
      partMax = *it;
    }
  }

  cleanup_();

  return std::make_pair(rMax, partMax);
}

void
PartitionTest::formPartitions_() {
  // This is slow, as it constrained to calculate k-size partitions from an 
  // algorithm that in unconstrained form calculates all partitions.

  std::cout << "THEORETICAL NUM PARTITONS: " << combinatorics::Bell_n_k(numElements_, T_) << std::endl;
  std::cout << "MONOTONIC/ALL: " << std::setprecision(4) << combinatorics::mon_to_all_ratio(numElements_, T_) << std::endl;
  std::cout << "COMPUTING PARTITIONGS...\n";

  std::vector<std::vector<int>> lists;
  std::vector<int> indexes(elements_.size(), 0);
  lists.emplace_back(std::vector<int>());
  lists[0].insert(lists[0].end(), elements_.begin(), elements_.end());
  
  int counter = -1;
  
  for(;;){
    counter += 1;
    if (lists.size() == T_) {
      fList_.emplace_back(lists);
    }
    
    int i,index;
    bool obreak = false;
    for (i=indexes.size()-1;; --i) {
      if (i<=0){
	obreak = true;
	break;
      }
      index = indexes[i];
      lists[index].erase(lists[index].begin() + lists[index].size()-1);
      if (lists[index].size()>0)
	break;
      lists.erase(lists.begin() + index);
    }
      if(obreak) break;
      
      ++index;
      if (index >= lists.size())
	lists.emplace_back(std::vector<int>());
      for (;i<indexes.size();++i) {
	indexes[i]=index;
	lists[index].emplace_back(elements_[i]);
	index=0;
      }
      
  }    
  
}

resultPair
PartitionTest::get_results() const {
  return optimalResult_;
}

bool
PartitionTest::assertOrdered(const resultPair& r) const {
  for (auto& list: r.second) {
    std::vector<int> v(list.size());
    std::adjacent_difference(list.begin(), list.end(), v.begin());    
    if (std::find_if(v.begin()+1, v.end(), [](int a){return a!=1;}) != v.end()) {
      return false;
    }
  }
  return true;
}

int
PartitionTest::numPartitions() const {
  return fList_.size();
}

void
PartitionTest::set_a(std::vector<double>&& a) {
  a_ = a;
}

void
PartitionTest::set_b(std::vector<double>&& b) { 
  b_ = b;
}

std::vector<std::vector<std::vector<int>>>
PartitionTest::get_partitions() const {
  return fList_;
}
