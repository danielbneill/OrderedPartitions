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

using resultPair = std::pair<double, std::vector<std::vector<int>>>;

class PartitionTest {
public:
  PartitionTest(std::vector<double>& a, 
		std::vector<double>& b, 
		int T, 
		double gamma) :
    a_(a),
    b_(b),
    T_(T),
    gamma_(gamma),
    numElements_(a.size())
  { init_(true); }
  PartitionTest(std::vector<double>& a, 
		std::vector<double>& b,
		int T,
		double gamma,
		std::vector<std::vector<std::vector<int>>> fList) :
    a_(a),
    b_(b),
    T_(T),
    gamma_(gamma),
    fList_(fList)
  { init_(false); }
		
  void runTest();
  resultPair get_results() const;
  void print_pair(const resultPair&) const;
  void print_partitions() const;
  bool assertOrdered(const resultPair&) const;
  int numPartitions() const;
  std::vector<std::vector<std::vector<int>>> get_partitions() const;
  void set_a(std::vector<double>&&);
  void set_b(std::vector<double>&&);

private:
  int numElements_;
  std::vector<double> a_;
  std::vector<double> b_;
  int T_;
  double gamma_;
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
  void sort_by_priority_(std::vector<double>&, std::vector<double>&);
};

#endif
