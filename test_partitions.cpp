#include <cstdlib>
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
  float gamma;
  bool integer_gamma = false;
  std::istringstream Nss(argv[1]), Tss(argv[2]), gammass(argv[3]);
  Nss >> N; Tss >> T; gammass >> gamma;

  double lower_limit_a = 0.;
  if (gamma - static_cast<int>(gamma) <= 0.) {
    integer_gamma = true;
    lower_limit_a = -1.;
  }

  std::random_device rnd_device;
  std::mt19937 mersenne_engine {rnd_device()};
  std::uniform_real_distribution<double> dista{lower_limit_a, 1.};
  std::uniform_real_distribution<double> distb{0., 1.0};

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
  PartitionTest pt{a, b, T, gamma};
  auto partitions = pt.get_partitions();

  pt.print_partitions();
  std::cout << "EMPIRCAL NUM PARTITIONS: " << partitions.size() << std::endl;
  
  while (true) {
    // Random input
    std::generate(a.begin(), a.end(), gena);
    std::generate(b.begin(), b.end(), genb);

    // Instantiate PartitionTest object with precalculated partitions
    PartitionTest pt{a, b, T, gamma, partitions};

    // Optimize
    pt.runTest();
    
    // Print out problematic case
    if (!pt.assertOrdered(pt.get_results())) {
      std::cerr << "EXCEPTION\n";
      std::cerr << "a   = [ ";
      for (auto& el : a)
	std::cout << std::setprecision(16) << el << " ";
      std::cerr << "]" << std::endl;

      std::cerr << "b   = [ ";
      for (auto& el : b)
	std::cerr << std::setprecision(16) << el << " ";
      std::cerr << "]" << std::endl;

      std::cerr << "a/b = [ ";
      for (size_t i=0; i<a.size(); ++i)
	std::cerr << std::setprecision(8) << a[i]/b[i] << " ";
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

