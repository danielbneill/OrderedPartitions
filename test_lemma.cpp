#include <cstdlib>
#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <functional>
#include <iterator>

#include "test_partitions.hpp"

template<typename T>
T abs_(T t) {
  return t > 0 ? t : -1*t;
}

template<typename T>
T min_(T s, T t) {
  return s < t ? s : t;
}

double F(double x, double y) {
  return std::pow(x,2)/y;
}

auto main(int argc, char **argv) -> int {

  std::random_device rnd_device;
  std::mt19937 mersenne_engine {rnd_device()};
  std::discrete_distribution<int> distdisc{-1, 1};
  std::uniform_real_distribution<double> distunif(0.1, 1.0);
  std::uniform_real_distribution<double> distX1(-10., 10.);
  std::uniform_real_distribution<double> distY1(0.1, 10.);
  std::uniform_real_distribution<double> dista(-1., 1.);

  auto genX1 = [&distX1, &mersenne_engine]() {
    return distX1(mersenne_engine);
  };

  auto genY1 = [&distY1, &mersenne_engine]() {
    return distY1(mersenne_engine);
  };

  auto genbeta = [&distunif, &mersenne_engine](double Y1) {
    return Y1 * distunif(mersenne_engine);
  };

  auto genb = [&distunif, &mersenne_engine](double Y1) {
    return Y1 * distunif(mersenne_engine);
  };

  auto gena = [&dista, &distdisc, &mersenne_engine](double X1, double Y1, double b) {
    return (-1 * X1) + static_cast<double>(distdisc(mersenne_engine)) * abs_(X1) * sqrt((Y1-b)/Y1);
  };
  
  auto genalpha = [&distunif, &distdisc, &mersenne_engine](double a, double b, double beta, double X1, double Y1) {
    double mu, M = beta * (a/b), epsilon = .00001;
    double p_m = static_cast<double>(distdisc(mersenne_engine));
    double den = distunif(mersenne_engine);
    mu = min_(M-epsilon, X1 + (p_m/den) * abs_(X1)*sqrt((Y1-beta)/Y1));
    return mu;
  };

  auto genY2 = [&distunif, &mersenne_engine](double b) {
    // Only constraint is that Y is larger than b
    return b/distunif(mersenne_engine);
  };

  auto genX2 = [&distunif, &distdisc, &mersenne_engine](double alpha, double beta, double a, double Y2) {
    double mu, p_m = static_cast<double>(distdisc(mersenne_engine));
    double dunif = distunif(mersenne_engine);
    double lowerbound = alpha/(-1. - sqrt((Y2+beta)/Y2));    
    return (1/dunif) * lowerbound;

  };

  while (true) {
    // max(A,C) = A, max(B,D) = D example
    double X1 = genX1();
    double Y1 = genY1();
    double beta = genbeta(Y1);
    double b = genb(Y1);
    double a = gena(X1, Y1, b);
    double alpha = genalpha(a, b, beta, X1, Y1);
    double Y2 = genY2(b);
    double X2 = genX2(alpha, beta, a, Y2);

    /*
    assert(Y1 > 0);
    assert(Y2 > 0);
    assert(b > 0);
    assert(beta > 0);
    assert((F(X1-alpha, Y1-beta)-F(X1,Y1)) > 0);
    assert((F(X1+a, Y1+b) - F(X1,Y1)) < 0);
    assert((F(X2+alpha,Y2+beta) - F(X2,Y2)) < 0);
    assert(((a/b) - (alpha/beta)) > 0);
    */
    
    if ((F(X2-a,Y2-b) - F(X2,Y2)) > 0) {

      // Additional conditions
      if (X1 > alpha) { // Not the constraint
	if ((X1-alpha)/(Y1-beta) >= (alpha/beta)) { // Not the constraint
	  if (Y1 > beta) { // Not the constraint
	    if (true) {
	      // if ((X2/Y2) <= (alpha/beta)) { // The constraint
	      
		assert((F(X2-a,Y2-b) - F(X2,Y2)) > 0);
		
		double test1 = F(X1-alpha+a,Y1-beta+b) - F(X1,Y1);
		double test2 = F(X2+alpha-a,Y2+beta-b) - F(X2,Y2);
		double test3 = (F(X1-alpha,Y1-beta)+F(X2+alpha,Y2+beta)) - (F(X1,Y1)+F(X2,Y2));
		double test4 = (F(X1+a,Y1+b)+F(X2-a,Y2-b)) - (F(X1,Y1)+F(X2,Y2));
		if (!((test1>0) || (test2>0) || (test3>0) || (test4>0))) {

		  std::cout << "F(X1-alpha+a,Y1-beta+b) - F(X1,Y1):                              " << test1 << std::endl;
		  std::cout << "F(X2+alpha-a,Y2+beta-b) - F(X2,Y2):                              " << test2 << std::endl;
		  std::cout << "(F(X1-alpha,Y1-beta)+F(X2+alpha,Y2+beta)) - (F(X2,Y2)+F(X2,Y2)): " << test3 << std::endl;
		  std::cout << "(F(X1+a,Y1+a)+F(X2-a,Y2-b)) - (F(X1,Y1)+F(X2,Y2)):               " << test4 << std::endl;

		  std::cout << "F(X1-alpha, Y1-beta) - F(X1,Y1): (>0)" << F(X1-alpha, Y1-beta) - F(X1,Y1) << std::endl;
		  std::cout << "F(X1+a, Y1+b) - F(X1,Y1):        (<0)" << F(X1+a, Y1+b) - F(X1,Y1) << std::endl;
		  std::cout << "F(X2+alpha, Y2+beta) - F(X2,Y2): (<0)" << F(X2+alpha, Y2+beta) - F(X2,Y2) << std::endl;
		  std::cout << "F(X2-a,Y2-b) - F(X2,Y2):         (>0)" << F(X2-a,Y2-b) - F(X2,Y2) << std::endl;
		  std::cout << "a/b - alpha/beta:                (>0)" << a/b - alpha/beta << std::endl;

		  std::cout << "X1 = " << X1 << std::endl;
		  std::cout << "Y1 = " << Y1 << std::endl;
		  std::cout << "X2 = " << X2 << std::endl;
		  std::cout << "Y2 = " << Y2 << std::endl;
		  std::cout << "beta = " << beta << std::endl;
		  std::cout << "alpha = " << alpha << std::endl;
		  std::cout << "b = " << b << std::endl;
		  std::cout << "a = " << a << std::endl;
		  
		  std::cout << "\n\n";
		}
	      }
	    }
	}
      }
    }
  }

  return 0;
}
