#include <list>
#include <vector>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <limits>
#include <iostream>

#include "graph.hpp"

float
PartitionGraph::compute_weight(int i, int j) {
  float weight = -1. * std::pow(std::accumulate(a_.begin()+i, a_.begin()+j, 0.), 2) /
    std::accumulate(b_.begin()+i, b_.begin()+j, 0.);
  return weight;
}

float
PartitionGraph::compute_weight(int i, int j, std::vector<float>& diffs) {
  return diffs[i*n_+j];
}

void 
PartitionGraph::add_edge_and_weight(int j, int k, std::vector<float> &&diffs)
{
  int curr_node = node_to_int(j, k);
  float weight;

  if (k==1) {
    // level 1, just connect source to node
    weight = compute_weight(0, j, diffs);
    auto r = boost::add_edge(0, curr_node, EdgeWeightProperty(weight), G_);
  }
  else if (k==T_) {
    // level T_, connect all nodes in previous layer to node
    for (size_t i=j; i<j+per_level_-1; ++i) {
      weight = compute_weight(i, k, diffs);
      auto r = boost::add_edge(node_to_int(i, k-1), curr_node, EdgeWeightProperty(weight), G_);
		      }
  }
  else {
    // level >= 2
    for (size_t i=k-1; i<j; ++i) {
      weight = compute_weight(i, j, diffs);
      auto r = boost::add_edge(node_to_int(i, k-1), curr_node, EdgeWeightProperty(weight), G_);
    }
  }
}

void 
PartitionGraph::create() 
{
  // partial sums for ease of weight calculation
  std::vector<float> asum(n_), bsum(n_);
  std::partial_sum(a_.begin(), a_.end(), asum.begin(), std::plus<float>());
  std::partial_sum(b_.begin(), b_.end(), bsum.begin(), std::plus<float>());

  // create sparse matrix of differences
  int numDiff = (n_+1)*(n_);
  float wt;
  std::vector<float> diffs = std::vector<float>(numDiff);
  for (size_t j=1; j<=n_; ++j) { diffs[j] = -1. * std::pow(asum[j-1], 2)/bsum[j-1]; }
  for (size_t i=1; i<=n_; ++i) {
    for (size_t j=i+1; j<=n_; ++j) {
      wt = -1. * std::pow(asum[j-1]-asum[i-1], 2)/(bsum[j-1]-bsum[i-1]);
      diffs[i*n_+j] = wt;
    }
  }
  std::cerr << "PRECOMPUTES COMPLETE\n";

  // source added implicitly via add_edge()
  
  // add layers for $T \in \{1, \dots T__{-1}\}$
  std::cerr << "CONSTRUCTING GRAPH\n";
  for(size_t k=1; k<T_; ++k) {
    for(size_t j=k; j<(k+per_level_); ++j) {
      add_edge_and_weight(j, k, std::move(diffs));
    }
    std::cerr << "  LAYER " << k << " OF " << T_-1 << " COMPLETE\n";
  }
  
  // add sink, must add explicitly
  float weight;
  int curr_node = (T_-1) * per_level_ + 1;
  for (size_t j=T_-1; j<n_; ++j) {
    weight = compute_weight(j, n_, diffs);
    auto r = boost::add_edge(node_to_int(j, T_-1), curr_node, EdgeWeightProperty(weight), G_);
  }
  std::cerr << "GRAPH CONSTRUCTION COMPLETE\n";
}

void
PartitionGraph::optimize() {
  // Glorified bellman-ford call
  std::cerr << "COMPUTING SHORTEST PATH\n";

  int nb_vertices = boost::num_vertices(G_);
  boost::property_map<graph_t, float EdgeWeightProperty::*>::type weight_pmap;
  weight_pmap = boost::get(&EdgeWeightProperty::weight, G_);
  
  // init the distance
  std::vector<float> distance(nb_vertices, (std::numeric_limits<float>::max)());
  distance[0] = 0.;

  // init the predecessors (identity function)
  std::vector<std::size_t> parent(nb_vertices);
  for (int i = 0; i < nb_vertices; ++i)
    parent[i] = i;

  // bellman-ford
  bool r = bellman_ford_shortest_paths(G_, 
				       nb_vertices, 
				       boost::weight_map(weight_pmap).
				       distance_map(&distance[0]).
				       predecessor_map(&parent[0])
				       );
  
  // optimal paths
  using ipair = std::pair<int,int>;
  std::list<int> pathlist;
  int first=0, index = parent.back();

  // int pathlist, in reverse order
  while (index > 0) { pathlist.push_back(index); index = parent[index]; }
  // node optimalpath
  std::for_each(pathlist.rbegin(), pathlist.rend(), [this, &first](int a){
		  ipair node = int_to_node(a);
		  int last = node.first;
		  this->optimalpath_.push_back(std::make_pair(first, last));
		  first = last;
		});
  optimalpath_.push_back(std::make_pair(first, n_));

  std::for_each(optimalpath_.begin(), optimalpath_.end(), [](ipair i){
		  std::cout << "[" << i.first << ", " << i.second << ") --> ";
		});
  std::cout << " >>SINK<< \n";

  /*
    std::cout << "WEIGHTS\n";
    std::for_each(optimalpath_.begin(), optimalpath_.end(), [this](ipair i) {
    std::cout << "[" << i.first << ", " << i.second << ") : " 
    << this->compute_weight(i.first, i.second) << "\n";
    });
  */

}

std::list<std::pair<int,int>>
PartitionGraph::get_optimal_path() const {
  return optimalpath_;
}

void
PartitionGraph::write_dot() const {
  write_graphviz(std::cout, G_);
}

std::vector<int>
PartitionGraph::get_optimal_path_extern() {
  // just flatten everything out
  std::vector<int> optimalpath;
  
  for (auto& node : optimalpath_) {
    optimalpath.push_back(node.first);
    optimalpath.push_back(node.second);
  }

  return optimalpath;
}
    
int
PartitionGraph::node_to_int(int i, int j) {
  // valid for levels 1 through (T-1)
  return (j-1)*per_level_ + (i-j+1);
}

std::pair<int,int> 
PartitionGraph::int_to_node(int m) {
  // valid for levels 1 through (T-1)
  int a = 1 + int((m-1)/per_level_);
  int b = m - (a-1)*per_level_ + a - 1;
  return std::make_pair(b, a);
}


