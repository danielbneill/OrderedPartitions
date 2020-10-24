#include <gtest/gtest.h>

#include <iostream>
#include <vector>
#include <random>
#include <iterator>
#include <algorithm>
#include <numeric>
#include <iterator>

#include "graph.hpp"
#include "DP.hpp"

void sort_by_priority(std::vector<float>& a, std::vector<float>& b) {
  std::vector<int> ind(a.size());
  std::iota(ind.begin(), ind.end(), 0);
  
  std::stable_sort(ind.begin(), ind.end(),
		   [&a, &b](int i, int j) {
		     return (a[i]/b[i]) < (a[j]/b[j]);
		   });
  std::vector<float> a_s, b_s;
  for (auto i : ind) {
    a_s.push_back(a[i]);
    b_s.push_back(b[i]);
  }

  std::copy(a_s.begin(), a_s.end(), a.begin());
  std::copy(b_s.begin(), b_s.end(), b.begin());
	    
}

void pretty_print_subsets(std::vector<std::vector<int>>& subsets) {
  std::cout << "SUBSETS\n";
  std::cout << "[\n";
  std::for_each(subsets.begin(), subsets.end(), [](std::vector<int>& subset){
		  std::cout << "[";
		  std::copy(subset.begin(), subset.end(),
			    std::ostream_iterator<int>(std::cout, " "));
		  std::cout << "]\n";
		});
  std::cout << "]" << std::endl;
}


TEST(PartitionGraphTest, Baselines) {

  std::vector<float> a{0.0212651 , -0.20654906, -0.20654906, -0.20654906, -0.20654906,
      0.0212651 , -0.20654906,  0.0212651 , -0.20654906,  0.0212651 ,
      -0.20654906,  0.0212651 , -0.20654906, -0.06581402,  0.0212651 ,
      0.03953075, -0.20654906,  0.16200014,  0.0212651 , -0.20654906,
      0.20296943, -0.18828341, -0.20654906, -0.20654906, -0.06581402,
      -0.20654906,  0.16200014,  0.03953075, -0.20654906, -0.20654906,
      0.03953075,  0.20296943, -0.20654906,  0.0212651 ,  0.20296943,
      -0.20654906,  0.0212651 ,  0.03953075, -0.20654906,  0.03953075};
  std::vector<float> b{0.22771114, 0.21809504, 0.21809504, 0.21809504, 0.21809504,
      0.22771114, 0.21809504, 0.22771114, 0.21809504, 0.22771114,
      0.21809504, 0.22771114, 0.21809504, 0.22682739, 0.22771114,
      0.22745816, 0.21809504, 0.2218354 , 0.22771114, 0.21809504,
      0.218429  , 0.219738  , 0.21809504, 0.21809504, 0.22682739,
      0.21809504, 0.2218354 , 0.22745816, 0.21809504, 0.21809504,
      0.22745816, 0.218429  , 0.21809504, 0.22771114, 0.218429  ,
      0.21809504, 0.22771114, 0.22745816, 0.21809504, 0.22745816};

  std::vector<std::vector<int>> expected = {
    {1, 2, 3, 4, 6, 8, 10, 12, 16, 19, 22, 23, 25, 28, 29, 32, 35, 38, 21}, 
    {13, 24}, 
    {0, 5, 7, 9, 11, 14, 18, 33, 36, 15, 27, 30, 37, 39}, 
    {17, 26}, 
    {20, 31, 34}
  };

  
  // sort_by_priority(a, b);

  auto pg = PartitionGraph(40, 5, a, b);
  auto opt = pg.get_optimal_subsets_extern();

  for (size_t i=0; i<expected.size(); ++i) {
    auto expected_subset = expected[i], opt_subset = opt[i];
    ASSERT_EQ(expected_subset.size(), opt_subset.size());
    for(size_t j=0; j<expected_subset.size(); ++j) {
      ASSERT_EQ(expected_subset[j], opt_subset[j]);
    }
  }
   
}

TEST(PartitionGraphTest, OrderedProperty) {
  // Case (n,T) = (50,5)
  int n = 50, T = 5;
  
  std::default_random_engine gen;
  gen.seed(std::random_device()());
  std::uniform_real_distribution<float> dist(1., 10.);

  std::vector<float> a(n), b(n);

  for (size_t i=0; i<5; ++i) {
    for (auto &el : a)
      el = dist(gen);
    for (auto &el : b)
      el = dist(gen);

    // Presort
    sort_by_priority(a, b);

    auto pg = PartitionGraph(n, T, a, b);
    auto opt = pg.get_optimal_subsets_extern();

    int sum;
    std::vector<int> v;

    for (auto& list : opt) {
      v.resize(list.size());
      std::adjacent_difference(list.begin(), list.end(), v.begin());
      sum = std::accumulate(v.begin()+1, v.end(), 0);
    }
    
    // We ignored the first element as adjacent_difference has unintuitive
    // result for first element
    ASSERT_EQ(sum, v.size()-1);
  }
  
}

TEST(DPSolverTest, Baselines ) {

  std::vector<float> a{0.0212651 , -0.20654906, -0.20654906, -0.20654906, -0.20654906,
      0.0212651 , -0.20654906,  0.0212651 , -0.20654906,  0.0212651 ,
      -0.20654906,  0.0212651 , -0.20654906, -0.06581402,  0.0212651 ,
      0.03953075, -0.20654906,  0.16200014,  0.0212651 , -0.20654906,
      0.20296943, -0.18828341, -0.20654906, -0.20654906, -0.06581402,
      -0.20654906,  0.16200014,  0.03953075, -0.20654906, -0.20654906,
      0.03953075,  0.20296943, -0.20654906,  0.0212651 ,  0.20296943,
      -0.20654906,  0.0212651 ,  0.03953075, -0.20654906,  0.03953075};
  std::vector<float> b{0.22771114, 0.21809504, 0.21809504, 0.21809504, 0.21809504,
      0.22771114, 0.21809504, 0.22771114, 0.21809504, 0.22771114,
      0.21809504, 0.22771114, 0.21809504, 0.22682739, 0.22771114,
      0.22745816, 0.21809504, 0.2218354 , 0.22771114, 0.21809504,
      0.218429  , 0.219738  , 0.21809504, 0.21809504, 0.22682739,
      0.21809504, 0.2218354 , 0.22745816, 0.21809504, 0.21809504,
      0.22745816, 0.218429  , 0.21809504, 0.22771114, 0.218429  ,
      0.21809504, 0.22771114, 0.22745816, 0.21809504, 0.22745816};

  std::vector<std::vector<int>> expected = {
    {1, 2, 3, 4, 6, 8, 10, 12, 16, 19, 22, 23, 25, 28, 29, 32, 35, 38, 21}, 
    {13, 24}, 
    {0, 5, 7, 9, 11, 14, 18, 33, 36, 15, 27, 30, 37, 39}, 
    {17, 26}, 
    {20, 31, 34}
  };

  
  // sort_by_priority(a, b);

  auto dp = DPSolver(40, 5, a, b);
  auto opt = dp.get_optimal_subsets_extern();

  for (size_t i=0; i<expected.size(); ++i) {
    auto expected_subset = expected[i], opt_subset = opt[i];
    ASSERT_EQ(expected_subset.size(), opt_subset.size());
    for(size_t j=0; j<expected_subset.size(); ++j) {
      ASSERT_EQ(expected_subset[j], opt_subset[j]);
    }
  }

}

TEST(DPSolverTest, OrderedProperty) {
  // Case (n,T) = (50,5)
  int n = 50, T = 5;
  
  std::default_random_engine gen;
  gen.seed(std::random_device()());
  std::uniform_real_distribution<float> dist(1., 10.);

  std::vector<float> a(n), b(n);

  for (size_t i=0; i<5; ++i) {
    for (auto &el : a)
      el = dist(gen);
    for (auto &el : b)
      el = dist(gen);

    // Presort
    sort_by_priority(a, b);

    auto dp = DPSolver(n, T, a, b);
    auto opt = dp.get_optimal_subsets_extern();

    int sum;
    std::vector<int> v;

    for (auto& list : opt) {
      v.resize(list.size());
      std::adjacent_difference(list.begin(), list.end(), v.begin());
      sum = std::accumulate(v.begin()+1, v.end(), 0);
    }

    // We ignored the first element as adjacent_difference has unintuitive
    // result for first element
    ASSERT_EQ(sum, v.size()-1);
  }
  
}


TEST(MultiSolver, SmallScaleTieouts) {

  int n = 40, T = 10;
  size_t NUM_CASES = 250;
  
  std::default_random_engine gen;
  gen.seed(std::random_device()());
  std::uniform_real_distribution<float> dista(-10., 10.);
  std::uniform_real_distribution<float> distb(0., 10.);
  
  std::vector<float> a(n), b(n), c(n);

  for (size_t i=0; i<NUM_CASES; ++i) {
    for (auto &el : a)
      el = dista(gen);
    for (auto &el : b)
      el = distb(gen);

    sort_by_priority(a, b);

    for(int i=0; i<n; ++i)
      c[i] = a[i]/b[i];

    auto pg = PartitionGraph(n, T, a, b);
    auto opt_pg = pg.get_optimal_subsets_extern();

    auto dp = DPSolver(n, T, a, b);
    auto opt_dp = dp.get_optimal_subsets_extern();

    ASSERT_EQ(opt_pg.size(), opt_dp.size());

    /*
      std::copy(a.begin(), a.end(), std::ostream_iterator<float>(std::cout, " "));
      std::cout << std::endl;
      std::copy(b.begin(), b.end(), std::ostream_iterator<float>(std::cout, " "));
      std::cout << std::endl;
      std::copy(c.begin(), c.end(), std::ostream_iterator<float>(std::cout, " " ));
      std::cout << std::endl;
      std::cout << "PG SOLVER\n";
      pretty_print_subsets(opt_pg);
      std::cout << "DP SOLVER\n";
      pretty_print_subsets(opt_dp);
      std::cout << "=====\n";
    */

    // Scores
    float pg_score_num = 0., pg_score_den = 0.;
    float dp_score_num = 0., dp_score_den = 0.;
    float pg_score = 0., dp_score = 0.;
    for (size_t i=0; i<opt_pg.size(); ++i) {
      for (size_t j=0; j<opt_pg[i].size(); ++j) {
	pg_score_num += a[opt_pg[i][j]];
	pg_score_den += b[opt_pg[i][j]];
	dp_score_num += a[opt_dp[i][j]];
	dp_score_den += b[opt_dp[i][j]];
      }
      pg_score += pg_score_num*pg_score_num/pg_score_den;
      dp_score += dp_score_num*dp_score_num/dp_score_den;
    }

    // std::cout << "Scores (PG, DP): " << pg_score << " : " << dp_score << std::endl;
    
    
    for (size_t i=0; i<opt_pg.size(); ++i) {
      for (size_t j=0; j<opt_pg[i].size(); ++j) {
	ASSERT_EQ(opt_pg[i][j], opt_dp[i][j]);
      }
    }

  
  }
}

TEST(MultiSolver, LargeScaleTieouts) {

  int n = 500, T = 15;
  size_t NUM_CASES = 5;
  
  std::default_random_engine gen;
  gen.seed(std::random_device()());
  std::uniform_real_distribution<float> dista(-10., 10.);
  std::uniform_real_distribution<float> distb(0., 10.);
  
  std::vector<float> a(n), b(n), c(n);

  for (size_t i=0; i<NUM_CASES; ++i) {
    for (auto &el : a)
      el = dista(gen);
    for (auto &el : b)
      el = distb(gen);

    sort_by_priority(a, b);

    for(int i=0; i<n; ++i)
      c[i] = a[i]/b[i];

    auto pg = PartitionGraph(n, T, a, b);
    auto opt_pg = pg.get_optimal_subsets_extern();

    auto dp = DPSolver(n, T, a, b);
    auto opt_dp = dp.get_optimal_subsets_extern();

    ASSERT_EQ(opt_pg.size(), opt_dp.size());

    /*
      std::copy(a.begin(), a.end(), std::ostream_iterator<float>(std::cout, " "));
      std::cout << std::endl;
      std::copy(b.begin(), b.end(), std::ostream_iterator<float>(std::cout, " "));
      std::cout << std::endl;
      std::copy(c.begin(), c.end(), std::ostream_iterator<float>(std::cout, " " ));
      std::cout << std::endl;
      std::cout << "PG SOLVER\n";
      pretty_print_subsets(opt_pg);
      std::cout << "DP SOLVER\n";
      pretty_print_subsets(opt_dp);
      std::cout << "=====\n";
    */

    // Scores
    float pg_score_num = 0., pg_score_den = 0.;
    float dp_score_num = 0., dp_score_den = 0.;
    float pg_score = 0., dp_score = 0.;
    for (size_t i=0; i<opt_pg.size(); ++i) {
      for (size_t j=0; j<opt_pg[i].size(); ++j) {
	pg_score_num += a[opt_pg[i][j]];
	pg_score_den += b[opt_pg[i][j]];
	dp_score_num += a[opt_dp[i][j]];
	dp_score_den += b[opt_dp[i][j]];
      }
      pg_score += pg_score_num*pg_score_num/pg_score_den;
      dp_score += dp_score_num*dp_score_num/dp_score_den;
    }

    // std::cout << "Scores (PG, DP): " << pg_score << " : " << dp_score << std::endl;
    
    
    for (size_t i=0; i<opt_pg.size(); ++i) {
      for (size_t j=0; j<opt_pg[i].size(); ++j) {
	ASSERT_TRUE((opt_pg[i][j] == opt_dp[i][j]) || (dp_score > pg_score));
      }
    }

  
  }
}

auto main(int argc, char **argv) -> int {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

