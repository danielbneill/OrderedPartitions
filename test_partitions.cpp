#include <iostream>
#include <random>
#include <algorithm>
#include <functional>

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

  std::vector<float> a(4);
  std::vector<float> b(4);
  
  unsigned long count = 0;

  while (true) {
    std::generate(a.begin(), a.end(), gena);
    std::generate(b.begin(), b.end(), genb);
    
    PartitionTest pt{a, b, 3};

    // pt.set_a(std::move(a));
    // pt.set_b(std::move(b));
    
    pt.runTest();
    if (!pt.assertOrdered(pt.get_results())) {
      pt.print_pair(pt.get_results());
    }

    count++;
    if (!(count%10000)) {
      std::cout << "COUNT: " << count << std::endl;
    }
  }
  
  return 0;

}

