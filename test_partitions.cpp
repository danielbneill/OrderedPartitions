#include <cstdlib>
#include <cmath>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <functional>
#include <iterator>

#include "test_partitions.hpp"

auto main(int argc, char **argv) -> int {

  int N, T;
  double gamma, delta;
  std::istringstream Nss(argv[1]), Tss(argv[2]), gammass(argv[3]), deltass(argv[4]);
  Nss >> N; Tss >> T; gammass >> gamma; deltass >> delta;

  double lower_limit_a, lower_limit_b = 0.;
  double upper_limit_a = 1., upper_limit_b = 1.;
  if ((gamma - static_cast<int>(gamma) <= 0.) &&
      (delta - static_cast<int>(delta) <= 0.)) {
    lower_limit_a = -1.;
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

    // Instantiate PartitionTest object with precalculated partitions
    PartitionTest pt{a, b, T, gamma, partitions, delta};

    // Optimize
    pt.runTest();
    
    // Print out problematic case
    if (!pt.assertOrdered(pt.get_results())) {
      auto a_sorted = std::move(pt.get_a()), b_sorted = std::move(pt.get_b());

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

      pt.print_pair(pt.get_results());

      exit(0);
    }

    count++;
    if (!(count%100)) {
      std::cout << "COUNT: " << count << std::endl;
    }
  }
  
  return 0;

}

