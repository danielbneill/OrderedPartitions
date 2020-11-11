#ifndef __TEST_OPTIMAL_PARTITION_HPP__
#define __TEST_OPTIMAL_PARTITION_HPP__

#include <cmath>
#include <vector>
#include <algorithm>
#include <utility>
#include <numeric>
#include <iterator>
#include <atomic>

#include "threadpool.hpp"
#include "threadsafequeue.hpp"

class Score {
public:
  static double power_(double a, double b, double gamma) {
    return std::pow(a, gamma)/b;
  }
  static double power_plus_c(double a, double b, double gamma) {
    return std::pow(a, gamma)/b + 1.0;
  }
  static double neg_log_(double a, double b) {
    return -std::log(a)/b;
  }
  static double expq_(double a, double b) {
    return std::exp(-a)/b;
  }
  static double log_(double a, double b) {
    return -1.*std::log(1+a);
  }
  static double log_prod_(double a, double b) {
    return -1.*std::log((1+a)*(1+b));
  }
  static double double_log_(double a, double b) {
    return std::log(std::log((1+a)*(1+b)));
  }
  static double exp_(double a, double b) {
    return std::exp((1-a)*(1+b));
  }
  static double power_sum_(double a, double b, double gamma) {
    return std::pow(a,gamma) + std::pow(b,gamma);
  }
  static double power_prod_(double a, double b, double gamma) {
    return std::pow(a,gamma)*std::pow(b,gamma);
  }
  static double score_summand_(double a, double b, double gamma) {
    double eta = -2.0;
    // return std::pow(a,gamma)*std::pow(b,eta);
    // return std::pow(a,gamma)*std::pow(b,-2.0);
    // return std::pow(a,3.0)*std::pow(b,-2.0);
    // return std::pow(a/b,a)*std::exp(b-a);
    // return std::pow(a,1.0);
    return std::pow(a,4.5455)*std::pow(b,-3.5455) + std::pow(a,6.15)*std::pow(b,-5.15);
  }
};

using resultPair = std::pair<double, std::vector<std::vector<int>>>;

class PartitionTest {
public:
  PartitionTest(std::vector<double>& a, 
		std::vector<double>& b, 
		int T, 
		double gamma,
		double delta=1.0) :
    a_(a),
    b_(b),
    T_(T),
    gamma_(gamma),
    delta_(delta),
    numElements_(a.size())
  { init_(true); }
  PartitionTest(std::vector<double>& a, 
		std::vector<double>& b,
		int T,
		double gamma,
		std::vector<std::vector<std::vector<int>>> fList,
		double delta=1.0) :
    a_(a),
    b_(b),
    T_(T),
    gamma_(gamma),
    delta_(delta),
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
  void set_T(int);
  std::vector<double> get_a() const;
  std::vector<double> get_b() const;

private:
  std::vector<double> a_;
  std::vector<double> b_;
  int T_;
  double gamma_;
  double delta_;
  int numElements_;
  std::vector<int> elements_;
  std::vector<resultPair> results_;
  resultPair optimalResult_;
  std::vector<std::vector<std::vector<int>>> fList_;
  ThreadsafeQueue<resultPair> results_queue_;
  std::vector<ThreadPool::TaskFuture<void>> v_;

  mutable std::atomic<bool> optimization_done_{false};

  void init_(bool);
  void cleanup_();
  resultPair optimize_(int, int);
  void formPartitions_();
  void sort_by_priority_(std::vector<double>&, std::vector<double>&, double);
};

#endif
