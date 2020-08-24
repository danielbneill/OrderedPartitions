#include "solver_timer.hpp"

auto main(int argc, char **argv) -> int {

  int n, T, stride;

  std::istringstream nss(argv[1]), Tss(argv[2]), stridess(argv[3]);
  nss >> n; Tss >> T; stridess >> stride;

  std::default_random_engine gen;
  gen.seed(std::random_device()());
  std::uniform_real_distribution<double> dista(-10., 10.);
  std::uniform_real_distribution<double> distb(1., 10.);

  using dur = std::chrono::high_resolution_clock::duration::rep;
  std::vector<std::vector<dur>> times(n+1);
  for (size_t i=0; i<=n; ++i)
    {
      times[i] = std::vector<dur>(T+1);
      for(size_t j=0; j<=T; ++j)
	times[i][j] = dur{0};
    }

  for (size_t sampleSize=5; sampleSize<=n; sampleSize+=stride) {
    for (size_t numParts=2; numParts<=T; ++numParts) {

      if (sampleSize > numParts) {

	std::vector<float> a(sampleSize), b(sampleSize);
	
	for (auto &el : a)
	  el = dista(gen);
	for (auto &el : b)
	  el = distb(gen);
	
	precise_timer timer;
	auto pg = PartitionGraph(sampleSize, numParts, a, b);
	auto et = timer.elapsed_time<unsigned int, std::chrono::microseconds>();
	std::cout << "(n,T) = (" << sampleSize << ", " << numParts << "): " 
		  << et
		  << std::endl;
	times[sampleSize][numParts] = et;
      }

    }
  }
  
  // dump
  for( const auto &v : times) {
    for( const auto &el : v)
      std::cout << el << " ";
    std::cout << std::endl;
  }
  
  return 0;
}
