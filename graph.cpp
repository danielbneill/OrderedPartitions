#include <list>
#include <vector>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <limits>
#include <iostream>

#include "graph.hpp"

template<typename T>
class TD;

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
    __attribute__((unused)) auto r = boost::add_edge(0, curr_node, EdgeWeightProperty(weight), G_);
  }
  else if (k==T_) {
    // level T_, connect all nodes in previous layer to node
    for (int i=j; i<j+per_level_-1; ++i) {
      weight = compute_weight(i, k, diffs);
      __attribute__((unused)) auto r = boost::add_edge(node_to_int(i, k-1), curr_node, EdgeWeightProperty(weight), G_);
		      }
  }
  else {
    // level >= 2
    for (int i=k-1; i<j; ++i) {
      weight = compute_weight(i, j, diffs);
      __attribute__((unused)) auto r = boost::add_edge(node_to_int(i, k-1), curr_node, EdgeWeightProperty(weight), G_);
    }
  }
}

void
PartitionGraph::sort_by_priority(std::vector<float>& a, std::vector<float>& b) {
  std::vector<int> ind(a.size());
  std::iota(ind.begin(), ind.end(), 0);
  
  std::stable_sort(ind.begin(), ind.end(),
		   [&a, &b](int i, int j) {
		     return (a[i]/b[i]) < (a[j]/b[j]);
		   });

  priority_sortind_ = ind;

  // Inefficient reordering
  std::vector<float> a_s, b_s;
  for (auto i : ind) {
    a_s.push_back(a[i]);
    b_s.push_back(b[i]);
  }
  
  std::copy(a_s.begin(), a_s.end(), a.begin());
  std::copy(b_s.begin(), b_s.end(), b.begin());
}

void 
PartitionGraph::create() 
{
  // sort vectors by priority function G(x,y) = x/y
  sort_by_priority(a_, b_);

  // partial sums for ease of weight calculation
  std::vector<float> asum(n_), bsum(n_);
  std::partial_sum(a_.begin(), a_.end(), asum.begin(), std::plus<float>());
  std::partial_sum(b_.begin(), b_.end(), bsum.begin(), std::plus<float>());

  // create sparse matrix of differences
  int numDiff = (n_+1)*(n_);
  float wt;
  std::vector<float> diffs = std::vector<float>(numDiff);
  for (int j=1; j<=n_; ++j) { diffs[j] = -1. * std::pow(asum[j-1], 2)/bsum[j-1]; }
  for (int i=1; i<=n_; ++i) {
    for (int j=i+1; j<=n_; ++j) {
      wt = -1. * std::pow(asum[j-1]-asum[i-1], 2)/(bsum[j-1]-bsum[i-1]);
      diffs[i*n_+j] = wt;
    }
  }
  // std::cerr << "PRECOMPUTES COMPLETE\n";

  // source added implicitly via add_edge()
  
  // add layers for $T \in \{1, \dots T__{-1}\}$
  // std::cerr << "CONSTRUCTING GRAPH\n";
  for(int k=1; k<T_; ++k) {
    for(int j=k; j<(k+per_level_); ++j) {
      add_edge_and_weight(j, k, std::move(diffs));
    }
    // std::cerr << "  LAYER " << k << " OF " << T_-1 << " COMPLETE\n";
  }
  
  // add sink, must add explicitly
  float weight;
  int curr_node = (T_-1) * per_level_ + 1;
  for (int j=T_-1; j<n_; ++j) {
    weight = compute_weight(j, n_, diffs);
    __attribute__((unused)) auto r = boost::add_edge(node_to_int(j, T_-1), curr_node, EdgeWeightProperty(weight), G_);
  }
  // std::cerr << "GRAPH CONSTRUCTION COMPLETE\n";
}

void
PartitionGraph::optimize() {
  // Glorified bellman-ford call
  // std::cerr << "COMPUTING SHORTEST PATH\n";

  int nb_vertices = boost::num_vertices(G_);
  boost::property_map<graph_t, float EdgeWeightProperty::*>::type weight_pmap_;
  weight_pmap_ = boost::get(&EdgeWeightProperty::weight, G_);
  
  // init the distance
  std::vector<float> distance(nb_vertices, (std::numeric_limits<float>::max)());
  distance[0] = 0.;

  // init the predecessors (identity function)
  std::vector<std::size_t> parent(nb_vertices);
  for (int i = 0; i < nb_vertices; ++i)
    parent[i] = i;

  // bellman-ford
  __attribute__((unused)) bool r = bellman_ford_shortest_paths(G_, 
							       nb_vertices, 
							       boost::weight_map(weight_pmap_).
							       distance_map(&distance[0]).
							       predecessor_map(&parent[0])
							       );
  
  // optimal paths
  std::list<int> pathlist;
  int first=0, index = parent.back();

  // int pathlist, in reverse order
  while (index > 0) { pathlist.push_back(index); index = parent[index]; }
  // node optimalpath
  std::for_each(pathlist.rbegin(), pathlist.rend(), [this, &first](int a){
		  std::pair<int, int> node = int_to_node(a);
		  int last = node.first;
		  this->optimalpath_.push_back(std::make_pair(first, last));
		  first = last;
		});
  optimalpath_.push_back(std::make_pair(first, n_));

  int subset_ind = 0;
  for (auto& node : optimalpath_) {
    subsets_[subset_ind]= std::vector<int>();
    for(int i=node.first; i<node.second; ++i) {
      subsets_[subset_ind].push_back(priority_sortind_[i]);
    }
    subset_ind++;
  }

  std::for_each(optimalpath_.begin(), optimalpath_.end(), [this](std::pair<int, int> i) {
		  this->optimalweight_ += this->compute_weight(i.first, i.second);
		});

  // Details
  /*
    std::for_each(optimalpath_.begin(), optimalpath_.end(), [](std::pair<int, int> i){
    std::cout << "[" << i.first << ", " << i.second << ") --> ";
    });
    std::cout << " >>SINK<< \n";
    
    std::cout << "SORTIND\n";
    std::copy(priority_sortind_.begin(), priority_sortind_.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << "\n";
    
    std::cout << "WEIGHTS\n";
    std::for_each(optimalpath_.begin(), optimalpath_.end(), [this](std::pair<int, int> i) {
    std::cout << "[" << i.first << ", " << i.second << ") : " 
    << this->compute_weight(i.first, i.second) << "\n";
    });
    
    std::cout << "SUBSETS\n";
    std::cout << "[\n";
    std::for_each(subsets_.begin(), subsets_.end(), [](std::vector<int>& subset){
    std::cout << "[";
    std::copy(subset.begin(), subset.end(),
    std::ostream_iterator<int>(std::cout, " "));
    std::cout << "]\n";
    });
    std::cout << "]";
  */
    
}

std::list<std::pair<int,int>>
PartitionGraph::get_optimal_path() const {
  return optimalpath_;
}

std::vector<std::vector<int>>
PartitionGraph::get_optimal_subsets_extern() const {
  return subsets_;
}

float
PartitionGraph::get_optimal_weight_extern() const {
  return optimalweight_;
}

template <class NameProp>
class name_label_writer {
public:
  name_label_writer(NameProp nameprop) : nameprop_(nameprop) {}
  template <class VertexOrEdge>
  void operator()(std::ostream& out, const VertexOrEdge& v) const {
    out << "[label=\"" << nameprop_[v] << "\"]";
  }
private:
  NameProp nameprop_;
};

template<typename WeightProp>
class weight_label_writer {
public:
  weight_label_writer(WeightProp weightprop) : weightprop_(weightprop) {}
  template<typename VertexOrEdge>
  void operator()(std::ostream& out, const VertexOrEdge& v) const {
    out << "[label=\"" << get(weightprop_, v) << "\", color=\"grey\"]";    
  }
private:
  WeightProp weightprop_;
};

void
PartitionGraph::write_dot() {

  int nb_vertices = boost::num_vertices(G_);
  std::vector<std::string> nameProp(nb_vertices);
  for(int i=0; i<nb_vertices; ++i) {
    auto p = int_to_node(i);
    nameProp[i] = "(" + std::to_string(p.first) + ", " + std::to_string(p.second) + ")";
  }

  auto weightProp = boost::get(&EdgeWeightProperty::weight, G_);
  using weightPropType = boost::adj_list_edge_property_map<boost::directed_tag, float, float&, unsigned long, EdgeWeightProperty, float EdgeWeightProperty::*>;

  // Full labeling
  write_graphviz(std::cout, G_, name_label_writer<std::vector<std::string>>(nameProp), weight_label_writer<weightPropType>(weightProp));

  // Only node name labeling
  // write_graphviz(std::cout, G_, name_label_writer(nameProp));

  // No labeling
  // write_graphviz(std::cout, G_);
}

std::vector<int>
PartitionGraph::get_optimal_path_extern() const {
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


