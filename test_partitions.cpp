#include <cstdlib>
#include <iostream>
#include <random>
#include <algorithm>
#include <functional>
#include <iterator>

#include "test_partitions.hpp"

auto main(int argc, char **argv) -> int {

  std::random_device rnd_device;
  std::mt19937 mersenne_engine {rnd_device()};
  std::uniform_real_distribution<float> dista{-10., 10.};
  std::uniform_real_distribution<float> distb{0., 10.};

  auto gena = [&dista, &mersenne_engine]() {
    return dista(mersenne_engine);
  };
  auto genb = [&distb, &mersenne_engine]() {
    return distb(mersenne_engine);
  };

  // std::vector<float> a(5);
  // std::vector<float> b(5);

  std::vector<float> a{-9.92934, -4.2602, -9.73161, -8.90641, -0.671412};
  std::vector<float> b{6.59367, 4.78854, 8.3982e-06, 6.8287, 1.29791};
  
  unsigned long count = 0;

  PartitionTest pt{a, b, 3};
  auto partitions = pt.get_partitions();
  std::cout << "NUM PARTITIONS: " << partitions.size() << std::endl;
  pt.print_partitions();
  
  while (true) {
    // std::generate(a.begin(), a.end(), gena);
    // std::generate(b.begin(), b.end(), genb);

    std::vector<float> a{-9.92934, -4.2602, -9.73161, -8.90641, -0.671412};
    std::vector<float> b{6.59367, 4.78854, 8.3982e-06, 6.8287, 1.29791};
    
    PartitionTest pt{a, b, 3, partitions};

    pt.runTest();
    if (!pt.assertOrdered(pt.get_results())) {
      std::copy(a.begin(), a.end(), std::ostream_iterator<float>(std::cout, " "));
      std::cout << "\n";
      std::copy(b.begin(), b.end(), std::ostream_iterator<float>(std::cout, " "));
      std::cout << "\n";
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

