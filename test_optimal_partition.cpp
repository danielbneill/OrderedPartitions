#include "test_optimal_partition.hpp"

void
PartitionTest::sort_by_priority_(std::vector<float>& a, std::vector<float>& b) {
  std::vector<int> ind(a_.size());
  std::iota(ind.begin(), ind.end(), 0);
  stable_sort(ind.begin(), ind.end(),
	      [&a, &b](int i, int j) {
		return (a[i]/b[i]) < (a[j]/b[j]);
	      });
  std::vector<float> a_s(a.size()), b_s(a.size());
  for (size_t i=0; i<a.size(); ++i) {
    a_s[i] = a[ind[i]];
    b_s[i] = b[ind[i]];
  }

  std::copy(a_s.begin(), a_s.end(), a.begin());
  std::copy(b_s.begin(), b_s.end(), b.begin());

  /*
    for (size_t i=0; i<a.size(); ++i) {
    std::cout << a[i] << " : " << b[i] << " : " << a[i]/b[i] << std::endl;
    }
  */
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
  std::cout << "ratio: " << p.first << " ";
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
PartitionTest::init_() {
  elements_ = std::vector<int>(numElements_);
  std::iota(elements_.begin(), elements_.end(), 0);
  formPartitions_();
}

void
PartitionTest::cleanup_() {
  // results_queue_.clear();
  results_.clear();
}

resultPair
PartitionTest::optimize_(int b, int e) {
  float rSum, paSum, pbSum;
  float rMax = std::numeric_limits<float>::min();
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
      rSum += std::pow(paSum, 2)/pbSum;
    }
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
    if (std::adjacent_find(v.begin()+1, v.end(), std::not_equal_to<>()) != v.end()) {
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
PartitionTest::set_a(std::vector<float>&& a) {
  a_ = a;
}

void
PartitionTest::set_b(std::vector<float>&& b) { 
  b_ = b;
}
