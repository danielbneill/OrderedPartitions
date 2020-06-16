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

  // Free simulation
  while (true) {
    double X1 = distX1(mersenne_engine);
    double Y1 = distY1(mersenne_engine);
    double X2 = distX1(mersenne_engine);
    double Y2 = distY1(mersenne_engine);
    double alpha = distX1(mersenne_engine);
    double beta = distunif(mersenne_engine) * Y1;
    double a = distX1(mersenne_engine);
    double b = distunif(mersenne_engine) * Y2;

    if ((F(X1-alpha, Y1-beta)-F(X1,Y1)) >= 0.) {      
      if ((F(X1+a, Y1+b) - F(X1,Y1)) <= 0.) {
	if ((F(X2+alpha,Y2+beta) - F(X2,Y2)) <= 0.) {
	  if ((F(X2-a,Y2-b) - F(X2,Y2)) >= 0.) {
	    if (((a/b) - (alpha/beta)) > 0.) {
	      if ((X1/Y1) >= (alpha/beta)) {
		if ((X2/Y2) <= (a/b)) {
		  if (((X1-alpha)/(Y1-beta) - (a/b)) > 0.) {
		    // Base constraints complete

		    double s1 = (X1-alpha)*(X1-alpha)*beta/(Y1*(Y1-beta));
		    double s2 = alpha*(alpha-2*X1)/Y1;
		    double s3 = (-(X2+alpha)*(X2+alpha)*beta)/(Y2*(Y2+beta));
		    double s4 = alpha*(alpha+2*X2)/Y2;

		    double t1 = -(X1+a)*(X1+a)*b/(Y1*(Y1+b));
		    double t2 = a*(a+2*X1)/Y1;
		    double t3 = b*(X2-a)*(X2-a)/(Y2*(Y2-b));
		    double t4 = a*(a-2*X2)/Y2;

		    double top_row = s1+s2+s3+s4;
		    double bot_row = t1+t2+t3+t4;

		    double plus_minus = F(X1-alpha+a, Y1-beta+b) + F(X2+alpha-a, Y2+beta-b) - F(X1,Y1) - F(X2,Y2);

		    double new_sum1 = F(X1-(X1-alpha),Y1-(Y1-beta)) + F(X2+(X1-alpha), Y2+(Y1-beta)) - F(X1,Y1) - F(X2,Y2);
		    double new_sum2 = F(X1+(X2-a), Y1+(Y2-b)) + F(X2-(X2-a), Y2-(Y2-b)) - F(X1,Y1) - F(X2,Y2);
		    
		    // if ((X1 * X2) >= 0.) { // CASE 1
		    //   if ( !((top_row >= 0.) || (bot_row >= 0.))) { // CASE 1
		    // if ((X1 * X2) <= 0.) { // Note that this CASE 2 doesn't work		    
		    // if (((X1 * X2) <= 0.) && (top_row <= 0.) && (bot_row <= 0.)) { // CASE 2
		    //   if ( !((new_sum1 >= 0.) || (new_sum2 >= 0.))) { // CASE 2

		    if (X1*X2 >= 0) {
		      if ((s2+s4<=0.) && (t2+t4<=0.)) {
			// if ( (!((top_row >= 0.) || (bot_row >= 0.))) && (X1 * X2 >= 0.)) { // works
			// if ( (!((new_sum1 >= 0.) || (new_sum2 >= 0.))) && (X1 * X2 <= 0.)) { // doesn't work
			// if (!((top_row >= 0.) || (bot_row >= 0.) || (new_sum1 >= 0.))) { // works
			// if (!((top_row >= 0.) || (bot_row >= 0.) || (new_sum2 >= 0.))) { // doesn't work		
			// if (!((new_sum1 >= 0.) || (new_sum2 >= 0.))) { // doesn't work
			// if (!((plus_minus >= 0.) || (new_sum2 >= 0.))) { // doesn't work
			// if (!( (new_sum1 >= 0.) || (new_sum2 >= 0.) || (plus_minus >= 0.))) { // doesn't work

			std::cout << "S2+s4: " << s2+s4 << std::endl;
			std::cout << "t2+t4: " << t2+t4 << std::endl;

			std::cout << "F(X1,Y1) - F(X2,Y2): " << F(X1,Y1) - F(X2,Y2) << std::endl;
			std::cout << "F(X1-alpha, Y1-beta) - F(X1,Y1): (>0)" << F(X1-alpha, Y1-beta) - F(X1,Y1) << std::endl;
			std::cout << "F(X1+a, Y1+b) - F(X1,Y1):        (<0)" << F(X1+a, Y1+b) - F(X1,Y1) << std::endl;
			std::cout << "F(X2+alpha, Y2+beta) - F(X2,Y2): (<0)" << F(X2+alpha, Y2+beta) - F(X2,Y2) << std::endl;
			std::cout << "F(X2-a,Y2-b) - F(X2,Y2):         (>0)" << F(X2-a,Y2-b) - F(X2,Y2) << std::endl;
			std::cout << "a/b - alpha/beta:                (>0)" << a/b - alpha/beta << std::endl;
			std::cout << "\n";
		    
			std::cout << "top_row: " << top_row << std::endl;
			std::cout << "bot_row: " << bot_row << std::endl;
			std::cout << "plus_minus: " << plus_minus << std::endl;
			std::cout << "new_sum1: " << new_sum1 << std::endl;
			std::cout << "new_sum2: " << new_sum2 << std::endl;
		    
			std::cout << "X1 = " << X1 << std::endl;
			std::cout << "Y1 = " << Y1 << std::endl;
			std::cout << "X2 = " << X2 << std::endl;
			std::cout << "Y2 = " << Y2 << std::endl;
			std::cout << "beta = " << beta << std::endl;
			std::cout << "alpha = " << alpha << std::endl;
			std::cout << "b = " << b << std::endl;
			std::cout << "a = " << a << std::endl;
			std::cout << "\n";		    
		      }
		    }
		  }
		}
	      }	    
	    }
	  }
	}
      }
    }
  }

  return 0;
}
