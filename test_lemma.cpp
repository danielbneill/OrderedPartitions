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

double max_(double a, double b) {
  return a > b ? a : b;
}

double min_(double a, double b) {
  return a < b ? a : b;
}

double max_vec(std::vector<double> v) {
  return *std::max_element(v.begin(), v.end());
}

double FAll(double X1, double Y1, double X2, double Y2, double alpha, double beta) {
  return std::pow(X1-alpha, 2)/(Y1-beta) + std::pow(X2+alpha, 2)/(Y2+beta) - std::pow(X1, 2)/Y1 - std::pow(X2, 2)/Y2;
}

auto main(int argc, char **argv) -> int {

  std::random_device rnd_device;
  std::mt19937 mersenne_engine {rnd_device()};
  std::discrete_distribution<int> distdisc{-1, 1};
  std::uniform_real_distribution<double> distunif(0.00001, 1.0);
  std::uniform_real_distribution<double> distX1(-1., 1.);
  std::uniform_real_distribution<double> distY1(0.00001, 100.);
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
		// if (((X1-alpha)/(Y1-beta)) >= (alpha/beta)) {
		// if (((X1+alpha)/(Y1+beta)) >= (alpha/beta)) {
	      if ((X2/Y2) <= (a/b)) {
	      // if (((X2-a)/(Y2-a)) <= (a/b)) {
	      // if (((X2+a)/(Y2+a)) <= (a/b)) {
	      if ((Y1 - beta) >= 0.) {
		if ((Y2 - b) >= 0.) {

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

			    double new_sum11 = F(X1-(X1-alpha), Y1-(Y1-beta))-F(X1,Y1);
			    double new_sum12 = F(X2+(X1-alpha), Y2+(Y1-beta))-F(X2,Y2);
					
			    // This works
			    // if ((new_sum11 <= 0.) || (new_sum12 <= 0.)) {
			    //   if (((s1+s2+s3+s4)<=0.) && ((t1+t2+t3+t4)<=0.)) {
		    
			    // This doesn't work, with s or t
			    // if (new_sum11 >= 0.) {
			    //   if ((t1+t2+t3+t4) <= 0.) {
			
			    // if ((t1+t2+t3+t4)<0.) { WORKS
			    //if ((s1+s2+s3+s4)<0.) { WORKS

			    if ((X1 >= 0.) && (X2 >= 0.)) {
					      
				// This doesn't work
				// if (((F(X1+(X2-a),Y1+(Y2-b))-F(X1,Y1)) <= 0.) || ((F(X2-(X2-a),Y2-(Y2-b))-F(X2,Y2)) <= 0.)) {
				//  if (((s1+s2+s3+s4)<=0.) && ((t1+t2+t3+t4)<=0.)) {
		    
				// This doesn't work
				// if (true) {
				//   if ((new_sum1 <=0.) && (new_sum2 <= 0.)) {
			

				// if ((X1 * X2) <= 0.) { // CASE 1
				//   if ( !((top_row >= 0.) || (bot_row >= 0.))) { // CASE 1
				// if ((X1 * X2) <= 0.) { // Note that this CASE 2 doesn't work		    
				// if (((X1 * X2) <= 0.) && (top_row <= 0.) && (bot_row <= 0.)) { // CASE 2
				// if ( !((new_sum1 >= 0.) || (new_sum2 >= 0.))) { // CASE 2
				// if ((F(X1-(X1-alpha), Y1-(Y1-beta))-F(X1,Y1)) <= 0.) {
				// if ((F(X2+(X1-alpha), Y2+(Y1-beta))-F(X2,Y2)) <= 0.) {

				// if ((X1 >= 0) && (X2 >= 0.)) {
				// if ((s2+s4<=0.) && (t2+t4<=0.)) {
				// if (((s2 >=0.) && (s2+s4 <= 0.)) || ((s4 >=0.) && (s2+s4 <= 0.))) {
				// if (((s1+s2+s3+s4)<=0.) && ((t1+t2+t3+t4)<=0.) && (plus_minus <=0.)) {
				// if ((t1+t2+t3+t4)<=0.) {
				// if (((s1+s2+s3+s4)<=0.) && ((t1+t2+t3+t4)<=0.)) {
				// if ( (!((top_row >= 0.) || (bot_row >= 0.))) && (X1 * X2 >= 0.)) { // works
				// if ( (!((new_sum1 >= 0.) || (new_sum2 >= 0.))) && (X1 * X2 <= 0.)) { // doesn't work
				// if (!((top_row >= 0.) || (bot_row >= 0.) || (new_sum1 >= 0.))) { // works
				// if (!((top_row >= 0.) || (bot_row >= 0.) || (new_sum2 >= 0.))) { // doesn't work		
				// if (!((new_sum1 >= 0.) || (new_sum2 >= 0.))) { // doesn't work
				// if (!((plus_minus >= 0.) || (new_sum2 >= 0.))) { // doesn't work
				// if (!( (new_sum1 >= 0.) || (new_sum2 >= 0.) || (plus_minus >= 0.))) { // doesn't work
				
				// if ((X1*X2) >= 0.) 
				//   std::cout << "******************** VIOLATION TYPE 0 *******************\n";
				// if ((new_sum1 <=0.) && (new_sum2 <=0.)) {
				// if (((s1+s2+s3+s4) <=0.) && ((t1+t2+t3+t4) <=0.)) {
				// if ((top_row<0.) && (bot_row<0.)) {
			      // if (((X1/Y1)-(X2/Y2)) <= 0.) {
			      // if (((X1/Y1)-(X2/Y2)) <= 0.) {
			      // if ((alpha<=0.) && ((s1 <= 0.) || (s2 <= 0.))) {
			      // if (((beta/Y1)*F(X1,Y1) - (beta/Y2)*F(X2,Y2)) <= 0.) {
			      // if ((F(X1,Y1)/F(X2,Y2) - (Y1/Y2)) <= 0.) {
			      // if (((X1/Y1) - (X2/Y2)) <= 0.) {

			      // if (((X1/Y1) - (a/b)) <= 0.) {
			      
			      // if (((alpha > 0 ) && (  (((-Y1*Y2)/(Y1+Y2))*((X1/Y1)-(X2/Y2))*((X1/Y1)-(X2/Y2))) +(s1+s3)) <= 0.)) {

			      // Basis for subcase
			      double max_alpha = min_(X1*(1-sqrt((Y1-beta)/Y1)), X2*(sqrt((Y2+beta)/Y2)-1));
			      double max_alpha_approx = X2*sqrt(beta/Y2);
			      double g = 2*(X2*Y1-X1*Y2)/(Y1*Y2);
			      double h = (Y1+Y2)/(Y1*Y2);				
			      double s1s3_bound = beta*(((X1*X1)/(Y1*Y1))-((X2*X2)/(Y2*Y2)));
			      double s1s3_bound_approx = beta*(((2*a*2*a)/(b*b))-((X2*X2)/(Y2*Y2)));
			      double s1s3_bound_approx_appox = beta*(((2*a*2*a)/(b*b))-((a*a)/(b*b)));
			      // if ((alpha > 0) && ((h*max_alpha*max_alpha + g*max_alpha + (s1+s3)) <= 0.)) { 
			      // if ((alpha > 0) && ((abs_(max_alpha)*abs_(max_alpha*h)) >= s1s3_bound)) { 
			      // if ((alpha > 0) && ((abs_(max_alpha_approx)*abs_(max_alpha_approx*h)) >= s1s3_bound)) { 
			      if ((alpha > 0) && ((abs_(max_alpha_approx)*abs_(max_alpha_approx*h)) >= s1s3_bound)) { // best so far

			      // if ((alpha > 0.) && ((abs_(g) <= abs_(max_alpha*h)) || (abs_(max_alpha*(g+max_alpha*h)) >= abs_(s1s3_bound)))) {
			      // if ((alpha > 0.) && (abs_(g+h*max_alpha) >= abs_(g))) {

			      // if ((alpha*(g+h*alpha)) <= -beta*(((X1*X1)/(Y1*Y1)) - ((X2*X2)/(Y2*Y2)))) {
			      double q_a = ((Y1/Y2)+(Y2/Y1))/((X2/Y2)-(X1/Y1));
			      double q_b = 2.;
			      double q_c = -beta*((X1/Y1)+(X2/Y2));
			      double theta = (Y1/Y2)+(Y2/Y1);

			      // if ((alpha > 0.) && (((((2/theta)-beta)*(X1/Y1) - ((2/theta)+beta)*(X2/Y2))) >= 0.)) {
				

			      // This summarizes things
			      // if ((alpha > 0.) && ((-((q_b*q_b)/(2*q_a)) + q_c) >= 0.)) {
			      std::cout << "TEST0: " << q_a << " : " << q_b << " : " << q_c << std::endl;
			      std::cout << "TEST1: " 
					<< -(q_b*q_b)/(2*q_a) + q_c << " : " 
					<< q_b*q_b - 4*(q_a)*(q_c) << " : " 
					<< alpha*(g+h*alpha) << std::endl;
			      std::cout << "TEST2: " << s2+s4 << " : " << -(s1+s3) << std::endl;
			      std::cout << "TEST3: " << s2+s4 << " : " << -s1s3_bound << std::endl;
				
				
			      // double lhs = (F(X2,Y2)/F(X1,Y1))*((2*Y1+Y2)/(Y2));
			      // double lhs_approx = (X2/(2*X1))*((2*Y1+Y2)/(Y2));
			      // double rhs = 1.;
			      // if ((alpha > 0.) && (lhs >= rhs)) {
			      
			      double X2Y2_bound = (X2*X2)/(Y2*Y2);
			      double lhs = X2Y2_bound*((beta+Y2)/beta) + X2Y2_bound;
			      double rhs = ((X1*X1)/(Y1*Y1));
			      // if ((alpha > 0) && ((2*Y1+Y2)/(Y2))*X2Y2_bound >= rhs) {
			      // if ((alpha > 0) && ((F(X2,Y2)/F(X1,Y1)) >= (b/(2*Y1+b)))) { 
			      // if ((alpha > 0) && (((a*X2*Y1)/(b*X1*X1)) >= (b/(2*Y1+b)))) {
			      double n = 1 - sqrt((Y2-b)/Y2);
			      double d = 1 + sqrt((Y1+b)/Y1);
			      double lhs_bound = (X2/X1)*(n/d);
			      // if ((alpha > 0.) && (lhs_bound >= (Y2/(2*Y1+Y2)))) {
			      // if ((alpha > 0.) && (lhs_bound >= (Y2/(2*Y1+Y2)))) {
			      // if ((alpha > 0.) && (lhs_bound >= (b/(2*Y1+b)))) {


				double f1 = F(X2,Y2)/F(X1,Y1);
				double f2 = (X2/X1)*((1-sqrt((Y2-b)/Y2))/(1+sqrt((Y1+b)/Y1)));
				double f3 = (b/(2*Y1+b));
				double lhs_test = (X2/(Y2*X1)*((1-sqrt((Y2-b)/(Y2)))/(sqrt((2*Y1+b)/(Y1)))));
				double rhs_test = 1/(2*Y1+b);			
				// if ((alpha > 0.) && (lhs_test >= rhs_test)) {
				
				// double lhs1 = (X2/Y2)*(1/X1)*sqrt(2*Y1+b);
				// double lhs2 = sqrt(Y1/Y2)*(sqrt(Y2)-sqrt(Y2-b));				
				// if ((alpha > 0.) && (lhs1*lhs2 >= 1.)) { // left off here
				  

			      // for X1 >= 0, X2 <= 0 case
			      // if ((alpha <= 0) && (s1+s2+s3+s4+t1+t2+t3+t4) <= 0.) {
			      // if ((((b - beta) >= 0.) && ((X1/Y1) - (X2/Y2)) <= 0.)) {

			      // if (((s1+s3) <= 0.) && ((t1+t3) <= 0.) && (alpha <= 0.)) {
			      // if (((s1+s3) >= 0.) && ((s1+s2+s3+s4) <= 0.) && (alpha <= 0.)) {
			      // if (((s1+s3) <= 0.) && ((t1+t3) <= 0.) && (alpha >= 0.)) {
			      // if (((s1+s3) >= 0.) && (((s1+s2+s3+s4) <= 0.) && ((t1+t2+t3+t4) <= 0.)) && (alpha >= 0.)) {

				// std::cout << "TEST1: " << -1*((1/Y2) - (1/Y1)) << std::endl;
				// std::cout << "TEST2: " << -1*(2*((X2/Y2) - (X1/Y1)) - 2*alpha*((1/Y2) - (1/Y1))) << std::endl;

				// almost
				// std::cout << "TEST1: " << (s1+s2+s3+s4) - (F(X1-alpha,Y1-beta)-F(X1,Y1)+((-beta/(Y2+beta))*F(X2,Y2))) << std::endl;
				// std::cout << "TEST2: " << (F(X1-alpha,Y1-beta)-F(X1,Y1)+((-beta/(Y2+beta))*F(X2,Y2))) << std::endl;

			      // if ((F(X1-alpha,Y1-beta)+F(X2+alpha,Y2+beta)-F(X1,Y1)-F(X2,Y2)) < 0.) {
			      // if ((F(X1+a,Y1+b)+F(X2-a  ,Y2-b)-F(X1,Y1)-F(X2,Y2)) < 0.) {

				double tmp1 = ((beta/(Y1-beta))*(F(X1,Y1) + s2));
				double tmp2 = ((-beta/(Y2+beta))*(F(X2,Y2) + s4));
				double tmp3 = F(X1-alpha,Y1) - F(X1,Y1);
				double tmp4 = F(X2+alpha,Y2)-F(X2,Y2);
				double alphaStar = (beta*(X1+X2)-(X2*Y1-X1*Y2))/(Y1+Y2);

				// std::cout << "TEST1: " << ((-Y1*Y2)/(Y1+Y2))*((X1/Y1)-(X2/Y2))*((X1/Y1)-(X2/Y2)) << " : " << -(s1+s3) << std::endl;
				// std::cout << "TEST2: " << h*alpha*alpha + g*alpha << " : " << -(s1+s3) << std::endl;
				// std::cout << "TEST3: " << h*max_alpha*max_alpha + g*max_alpha + (s1+s3) << std::endl;
				// std::cout << "TEST4: " << abs_(alpha) << " : " << abs_(alpha*h + g) << " : " << -(s1+s3) << std::endl;
				// std::cout << "TEST: " << (beta/(Y1-beta))*F(X1,Y1) - (beta/(Y2+beta))*F(X2,Y2) << " : " << s1+s3 << std::endl;
				// std::cout << "TEST: " << FAll(X1,Y1,X2,Y2,alphaStar,beta) << std::endl;
				// std::cout << "TEST: " << s1+s2+s3+s4 << " : " << FAll(X1,Y1,X2,Y2,alpha,beta) << std::endl;
				// std::cout << "TEST: " << ((X1-alpha)*(X1-alpha)/(Y1-beta)) + ((X2+alpha)*(X2+alpha)/(Y2+beta)) - (X1*X1)/Y1 - (X2*X2)/Y2 << std::endl;
				// std::cout << "TEST: " << s1+s2 << " : " << ((X1-alpha)*(X1-alpha)/(Y1-beta)) - (X1*X1/Y1) << std::endl;
				// std::cout << "TEST: " << s3+s4 << " : " << ((X2+alpha)*(X2+alpha)/(Y2+beta)) - (X2*X2/Y2) << std::endl;
				// std::cout << "TEST: " << s1+s2+s3+s4 << " : " << (-beta/(Y2+beta)) + (alpha*(2*X2+alpha)/Y2) << std::endl;

				// std::cout << "TEST: " << t1+t2+t3+t4 << " : " << F(X1+a,Y1+b)+F(X2-a,Y2-b)-F(X1,Y1)-F(X2,Y2) << std::endl;
				std::cout << "******************** VIOLATION TYPE 1 *******************\n";
				std::cout << "s1: " << s1 << std::endl;
				std::cout << "s2: " << s2 << std::endl;
				std::cout << "s3: " << s3 << std::endl;
				std::cout << "s4: " << s4 << std::endl;
				std::cout << "t1: " << t1 << std::endl;
				std::cout << "t2: " << t2 << std::endl;
				std::cout << "t3: " << t3 << std::endl;
				std::cout << "t4: " << t4 << std::endl;
			
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
				}
				  // if (new_sum11 <= 0.)
				  // std::cout << "******************** VIOLATION TYPE 2 *******************\n";
				  // if (new_sum12 <= 0.)
				  // std::cout << "******************** VIOLATION TYPE 3 *******************\n";
				  // if (plus_minus <= 0.)
				  //  std::cout << "******************** VIOLATION TYPE 4 ******************* " << plus_minus << std::endl;

				/*
				std::cout << "s1: " << s1 << std::endl;
				std::cout << "s2: " << s2 << std::endl;
				std::cout << "s3: " << s3 << std::endl;
				std::cout << "s4: " << s4 << std::endl;
				std::cout << "t1: " << t1 << std::endl;
				std::cout << "t2: " << t2 << std::endl;
				std::cout << "t3: " << t3 << std::endl;
				std::cout << "t4: " << t4 << std::endl;
			
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
				*/
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
//}
//}
//}
//}
  return 0;
}
