#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <functional>
#include <iterator>

#include "test_partitions.hpp"

auto main(int argc, char **argv) -> int {

  std::random_device rnd_device;
  std::mt19937 mersenne_engine {rnd_device()};
  std::uniform_real_distribution<double> dista{-10., 10.};
  std::uniform_real_distribution<double> distb{0., 10.};

  auto gena = [&dista, &mersenne_engine]() {
    return dista(mersenne_engine);
  };
  auto genb = [&distb, &mersenne_engine]() {
    return distb(mersenne_engine);
  };

  std::vector<double> a(4);
  std::vector<double> b(4);

  unsigned long count = 0;

  PartitionTest pt{a, b, 3};
  auto partitions = pt.get_partitions();
  std::cout << "NUM PARTITIONS: " << partitions.size() << std::endl;
  pt.print_partitions();
  
  while (true) {
    std::generate(a.begin(), a.end(), gena);
    std::generate(b.begin(), b.end(), genb);

    PartitionTest pt{a, b, 3, partitions};

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
    if (!(count%100000)) {
      std::cout << "COUNT: " << count << std::endl;
    }
  }
  
  return 0;

}

