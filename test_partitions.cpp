#include <cstdlib>
#include <cmath>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>

#include "test_partitions.hpp"

template<class C, typename T>
bool contains_no(const C& c, const T& value) {
  return std::find(c.begin(), c.end(), value) == c.end();
}

template<class C, typename T, typename... Ts>
bool contains_no(const C& c, const T a, Ts... args) {
  return contains_no(c, a) && contains_no(c, args...);
}

double round(double a, unsigned places) {
  double rad = std::pow(10., places);
  return std::ceil(a * rad)/rad;
}

bool assertMixedSign(const std::vector<double>& a) {
  bool hasPos = std::find_if(a.begin(), a.end(), [](double x){ return x > 0;}) != a.end();
  bool hasNeg = std::find_if(a.begin(), a.end(), [](double x){ return x < 0;}) != a.end();
  return hasPos && hasNeg;
}

auto main(int argc, char **argv) -> int {

  // If no strongly consecutive partition exists, reduce T
  // and look for weakly consecutive ones
  const bool TEST_STRONGLY_CONSECUTIVE = false;

  int N, T;
  double gamma, delta;
  std::istringstream Nss(argv[1]), Tss(argv[2]), gammass(argv[3]), deltass(argv[4]);
  Nss >> N; Tss >> T; gammass >> gamma; deltass >> delta;

  bool aMixedSign = false;
  if (argc > 5) {
    std::istringstream aMixedSignss(argv[5]);
    aMixedSignss >> aMixedSign;
  }

  double lower_limit_a = 0., upper_limit_a = 1.;
  double lower_limit_b = 0., upper_limit_b = 1.;

  if ((gamma - std::floor(gamma) <= 0.) &&
      (delta - std::floor(delta) <= 0.)) {
    lower_limit_a = -1.;
  }

  if (aMixedSign && !(lower_limit_a < 0)) {
    std::cerr << "aMixedSign incompatible with lower limit for a" << std::endl;
    return -1;
  }

  std::random_device rnd_device;
  std::mt19937 mersenne_engine {rnd_device()};
  std::uniform_real_distribution<double> dista{lower_limit_a, upper_limit_a};
  std::uniform_real_distribution<double> distb{lower_limit_b, upper_limit_b};

  auto gena = [&dista, &mersenne_engine]() {
    return dista(mersenne_engine);
  };
  auto genb = [&distb, &mersenne_engine]() {
    return distb(mersenne_engine);
  };

  std::vector<double> a(N);
  std::vector<double> b(N);

  unsigned long count = 0;

  // Instantiate PartitionTest to precalculate partitions
  PartitionTest pt{a, b, T, gamma, delta};
  auto partitions = pt.get_partitions();

  pt.print_partitions();
  std::cout << "EMPIRCAL NUM PARTITIONS: " << partitions.size() << std::endl;
  
  while (true) {
    // Random input
    std::generate(a.begin(), a.end(), gena);
    std::generate(b.begin(), b.end(), genb);

    if (aMixedSign) {
      while (not assertMixedSign(a)) {
	std::generate(a.begin(), a.end(), gena);
      }
    }

    // Instantiate PartitionTest object with precalculated partitions
    PartitionTest pt{a, b, T, gamma, partitions, delta};

    // Optimize
    pt.runTest();

    // Print out problematic case    
    if (!pt.assertOrdered(pt.get_results())) {
      // Replay everything
      std::vector<std::unique_ptr<PartitionTest>> pt_vec(T+1);
      std::vector<bool> isConsecutive(T+1, false);
      for (int i=T; i>=1; --i) {
	pt_vec[i] = std::make_unique<PartitionTest>(a, b, i, gamma, delta);
      }
      
      // Successively check smaller partition sizes
      if (TEST_STRONGLY_CONSECUTIVE) {
	for(size_t i=T; i>= 1; --i) {
	  pt_vec[i]->runTest();
	  if (!pt_vec[i]->assertOrdered(pt_vec[i]->get_results()))
	    continue;
	  else if (i > 1)
	    isConsecutive[i] = true;
	}
	// XXX
	// Look at all cases <= T
	if (contains_no(isConsecutive, true)) {
	// if (true) {
	  auto a_sorted = pt.get_a(), b_sorted = pt.get_b();
	  
	  std::cerr << "EXCEPTION: gamma = " << gamma << " delta = " << delta << std::endl;
	  std::cerr << "a   = [ ";
	  for (auto& el : a_sorted)
	    std::cout << std::setprecision(16) << el << " ";
	  std::cerr << "]" << std::endl;
	  
	  std::cerr << "b   = [ ";
	  for (auto& el : b_sorted)
	    std::cerr << std::setprecision(16) << el << " ";
	  std::cerr << "]" << std::endl;
	  
	  std::cerr << "a^delta/b = [ ";
	  for (size_t i=0; i<a_sorted.size(); ++i)
	    std::cerr << std::setprecision(8) << pow(a_sorted[i], delta)/b_sorted[i] << " ";
	  std::cerr << "]" << std::endl;
	  
	  for (size_t i=T; i>=1; --i) {
	    std::cerr << "Maximal " << i << "-partition" << std::endl;
	    pt_vec[i]->print_pair(pt_vec[i]->get_results());
	    std::cerr << std::endl;
	  }
	  
	  exit(0);
	}
      }
      else {
	auto a_sorted = pt.get_a(), b_sorted = pt.get_b();

	std::cerr << "EXCEPTION: gamma = " << gamma << " delta = " << delta << std::endl;
	std::cerr << "a   = [ ";
	for (auto& el : a_sorted)
	  std::cout << std::setprecision(16) << el << " ";
	std::cerr << "]" << std::endl;
	
	std::cerr << "b   = [ ";
	for (auto& el : b_sorted)
	  std::cerr << std::setprecision(16) << el << " ";
	std::cerr << "]" << std::endl;
	
	std::cerr << "a^delta/b = [ ";
	for (size_t i=0; i<a_sorted.size(); ++i)
	    std::cerr << std::setprecision(8) << pow(a_sorted[i], delta)/b_sorted[i] << " ";
	std::cerr << "]" << std::endl;	

	std::cerr << "Maximal partition" << std::endl;
	pt.print_pair(pt.get_results());
	std::cerr << std::endl;

	exit(0);
      }
    }
      
    count++;
    if (!(count%1)) {
      std::cout << "COUNT: " << count << std::endl;
    }  
  }
  
  return 0;

}
