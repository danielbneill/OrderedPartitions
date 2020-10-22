#include <stdio.h>
#include <math.h>

#include <iostream>
#include <iomanip>

#include <boost/multiprecision/gmp.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

using boost::multiprecision::cpp_dec_float_100;
using namespace boost::multiprecision;

auto main() -> int {

  // Underflow example

  // Case 2
  // [156.3404907  274.61774855]
  // [41.89509207 73.5902494 ]
  // Case 3
  // [215.09921685 274.43260945]
  // [227.88630852 290.74691492]

  const double a_cases[] = {-769.88725716, -261.39267287, 156.3404907,  274.61774855, 215.09921685, 274.43260945};
  const double b_cases[] = {435.20909316, 147.76250352, 41.89509207, 73.5902494, 227.88630852, 290.74691492};
  const int num_cases = sizeof(a_cases)/2./sizeof(double);

  std::cout << "num_cases: " << num_cases << std::endl;
  
  for (size_t case_num=0; case_num<num_cases; ++case_num) {

    std::cout << "CASE: " << case_num << std::endl;

    // Using C++ native 64-bit types
    const double a[2] = {a_cases[case_num*2], a_cases[1+case_num*2]};
    const double b[2] = {b_cases[case_num*2], b_cases[1+case_num*2]};
    
    std::cout << "\n\nC++ native double type:\n";
    std::cout << "======================\n";
    std::cout << "a: ";
    std::cout << std::setprecision(std::numeric_limits<cpp_dec_float_100>::digits10)
	      << std::setw(50)
	      << a[0] << " " << a[1]
	      << std::endl;
    
    std::cout << "b: ";
    std::cout << std::setprecision(std::numeric_limits<cpp_dec_float_100>::digits10)
	      << std::setw(50)
	      << b[0] << " " << b[1]
	      << std::endl;
    
    std::cout << "\npriority, tau=1: \n";
    std::cout << std::setprecision(std::numeric_limits<cpp_dec_float_100>::digits10)
	      << "a[0]/b[0]: " << a[0]/b[0] << "\na[1]/b[1]: " << a[1]/b[1] << "\n"
	      << "a[0]/b[0] < a[1]/b[1]: " << (a[0]/b[0] < a[1]/b[1])
	      << std::endl;
    
    const double score_of_sum_double = pow(a[0]+ a[1], 2.0)/(b[0] + b[1]);
    const double sum_of_scores_double = pow(a[0], 2.0)/b[0] + pow(a[1], 2.0)/b[1];
    
    // We should have score_of_sum <= sum_of_scores, however...
    std::cout << std::setprecision(std::numeric_limits<cpp_dec_float_100>::digits10)
	      << "\nscore_of_sum_double:  " << score_of_sum_double << "\n" 
	      << "sum_of_scores_double: " << sum_of_scores_double << "\n"
	      << "score_of_sum_double < sum_of_scores_double: " << (score_of_sum_double < sum_of_scores_double)
	      << std::endl;
    
    
    // Using boost GMP wrapper
    const cpp_dec_float_100 a0[2] = {a[0], a[1]};
    const cpp_dec_float_100 b0[2] = {b[0], b[1]};
    // const cpp_dec_float_100 a0[2] = {-769.88725716, -261.39267287};
    // const cpp_dec_float_100 b0[2] = {435.20909316, 147.76250352};
    
    std::cout << "\nGMP multiprecision float_100 type:\n";
    std::cout << "======================\n";
    std::cout << "a0: ";
    std::cout << std::setprecision(std::numeric_limits<cpp_dec_float_100>::digits10)
	      << std::setw(50)
	      << a0[0] << " " << a0[1]
	      << std::endl;
    
    std::cout << "b0: ";
    std::cout << std::setprecision(std::numeric_limits<cpp_dec_float_100>::digits10)
	      << std::setw(50)
	      << b0[0] << " " << b0[1]
	      << std::endl;
    
    std::cout << "\npriority, tau=1: \n";
    std::cout << std::setprecision(std::numeric_limits<cpp_dec_float_100>::digits10)
	      << "a0[0]/b0[0]: " << a0[0]/b0[0] << "\na0[1]/b0[1]: " << a0[1]/b0[1] << "\n"
	      << "a0[0]/b0[0] < a0[1]/b0[1]: " << (a0[0]/b0[0] < a0[1]/b0[1])
	      << std::endl;
    
    const cpp_dec_float_100 score_of_sum_multi((a0[0] + a0[1])*(a0[0] + a0[1])/(b0[0] + b0[1]));
    const cpp_dec_float_100 sum_of_scores_multi((a0[0] * a0[0])/b0[0] + (a0[1] * a0[1])/b0[1]);
    
    // We should have score_of_sum <= sum_of_scores, however...
    std::cout << std::setprecision(std::numeric_limits<cpp_dec_float_100>::digits10)
	      << "\nscore_of_sum_multi:  " << score_of_sum_multi << "\n" 
	      << "sum_of_scores_multi: " << sum_of_scores_multi << "\n"
	      << "score_of_sum_multi < sum_of_scores_multi: " << (score_of_sum_multi < sum_of_scores_multi)
	      << std::endl << std::endl;
  }
    
  return 0;
}
