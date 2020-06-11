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
  std::istringstream Nss(argv[1]), Tss(argv[2]), gammass(argv[3]);
  Nss >> N; Tss >> T; gammass >> gamma;

  std::random_device rnd_device;
  std::mt19937 mersenne_engine {rnd_device()};
  std::uniform_real_distribution<double> dista{-10000., 10000.};
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

  PartitionTest pt{a, b, T, gamma};
  auto partitions = pt.get_partitions();
  pt.print_partitions();
  std::cout << "EMPIRCAL NUM PARTITIONS: " << partitions.size() << std::endl;
  
  while (true) {
    std::generate(a.begin(), a.end(), gena);
    std::generate(b.begin(), b.end(), genb);

    PartitionTest pt{a, b, T, gamma, partitions};

    pt.runTest();
    if (!pt.assertOrdered(pt.get_results())) {
      for (auto& el : a)
	std::cout << std::setprecision(16) << el << " ";
      std::cout << std::endl;
      for (auto& el : b)
	std::cout << std::setprecision(16) << el << " ";
      std::cout << std::endl;
      pt.print_pair(pt.get_results());
      exit(0);
    }

    count++;
    if (!(count%1'000'000)) {
      std::cout << "COUNT: " << count << std::endl;
    }
  }
  
  return 0;

}

