#include <stdexcept>

#include "solver_timer.hpp"

auto main(int argc, char **argv) -> int {

  int n, T, stride, partsStride;

  if (argc < 5)
    throw std::invalid_argument("Must call with 4 input arguments.");

  std::istringstream nss(argv[1]), Tss(argv[2]), stridess(argv[3]);
  std::istringstream partsStridess(argv[4]);
  nss >> n; Tss >> T; stridess >> stride; partsStridess >> partsStride;

  std::default_random_engine gen;
  gen.seed(std::random_device()());
  std::uniform_real_distribution<double> dista(-10., 10.);
  std::uniform_real_distribution<double> distb(1., 10.);

  using dur = std::chrono::high_resolution_clock::duration::rep;
  std::vector<std::vector<dur>> times(n+1);
  for (int i=0; i<=n; ++i)
    {
      times[i] = std::vector<dur>(T+1);
      for(int j=0; j<=T; ++j)
	times[i][j] = dur{0};
    }

  for (int sampleSize=5; sampleSize<=n; sampleSize+=stride) {
    // for (int numParts=2; numParts<=T; numParts+=partsStride) {
    for (int numParts=T;numParts<=T;++numParts) {

      if (sampleSize > numParts) {

	std::vector<float> a(sampleSize), b(sampleSize);
	
	for (auto &el : a)
	  el = dista(gen);
	for (auto &el : b)
	  el = distb(gen);
	
	precise_timer timer;
	// auto pg = PartitionGraph(sampleSize, numParts, a, b);
	auto dp = DPSolver(sampleSize, numParts, a, b);
	auto et = timer.elapsed_time<unsigned int, std::chrono::microseconds>();
	std::cout << "(n,T) = (" << sampleSize << ", " << numParts << "): " 
	 	  << et
		  << std::endl;
	times[sampleSize][numParts] = et;
      }

    }
  }
  
  // dump
  auto delim = ",";
  for( const auto &v : times) {
    for( const auto &el : v)
      std::cout << el << delim;
    std::cout << std::endl;
  }
  
  return 0;
}
