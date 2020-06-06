#ifndef __TEST_OPTIMAL_PARTITION_HPP__
#define __TEST_OPTIMAL_PARTITION_HPP__

#include <cmath>
#include <vector>
#include <algorithm>
#include <utility>
#include <numeric>
#include <iterator>

#include "threadpool.hpp"
#include "threadsafequeue.hpp"

using resultPair = std::pair<float, std::vector<std::vector<int>>>;

class PartitionTest {
public:
  PartitionTest(std::vector<float>& a, std::vector<float>& b, int T) :
    a_(a),
    b_(b),
    T_(T),
    numElements_(a.size())
  { init_(true); }
  PartitionTest(std::vector<float>& a, 
		std::vector<float>& b,
		int T,
		std::vector<std::vector<std::vector<int>>> fList) :
    a_(a),
    b_(b),
    T_(T),
    fList_(fList)
  { init_(false); }
		
  void runTest();
  resultPair get_results() const;
  void print_pair(const resultPair&) const;
  void print_partitions() const;
  bool assertOrdered(const resultPair&) const;
  int numPartitions() const;
  std::vector<std::vector<std::vector<int>>> get_partitions() const;
  void set_a(std::vector<float>&&);
  void set_b(std::vector<float>&&);

private:
  int numElements_;
  std::vector<float> a_;
  std::vector<float> b_;
  int T_;
  std::vector<int> elements_;
  std::vector<resultPair> results_;
  resultPair optimalResult_;
  std::vector<std::vector<std::vector<int>>> fList_;
  ThreadsafeQueue<resultPair> results_queue_;
  std::vector<ThreadPool::TaskFuture<void>> v_;

  void init_(bool);
  void cleanup_();
  resultPair optimize_(int, int);
  void formPartitions_();
  void sort_by_priority_(std::vector<float>&, std::vector<float>&);
};

#endif
