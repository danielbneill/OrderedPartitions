#ifndef __GRAPH_HPP__
#define __GRAPH_HPP__

#include <list>
#include <utility>
#include <vector>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/topological_sort.hpp>
#include <boost/graph/bellman_ford_shortest_paths.hpp>
#include <boost/graph/graphviz.hpp>

//
// boost adjacency_list type
// adjacency_list<OutEdgeList, VertexList, Directed,
//                VertexProperties, EdgeProperties,
//                GraphProperites, EdgeList>
//
             
struct EdgeWeightProperty {
  EdgeWeightProperty(float w) : weight(w) {}
  float weight;
};

using graph_t = boost::adjacency_list<boost::listS,
				      boost::vecS,
				      boost::directedS,
				      boost::no_property,
				      EdgeWeightProperty>;

using edge_descriptor = boost::graph_traits<graph_t>::edge_descriptor;
using edge_iterator =   boost::graph_traits<graph_t>::edge_iterator;

using ipair = std::pair<int, int>;
using ipairlist = std::list<ipair>;
using ilist = std::list<int>;
using fvec = std::vector<float>;

class PartitionGraph {
public:
  PartitionGraph(int n, 
		 int T,
		 fvec a,
		 fvec b
		 ) :
    n_{n},
    T_{T},
    a_{a},
    b_{b},
    per_level_{n-T+1}
  { _init(); }

  PartitionGraph(int n,
		 int T,
		 float *a,
		 float *b
		 ):
    n_{n},
    T_{T},
    per_level_{n-T+1}
  { 
    a_.assign(a, a+n);
    b_.assign(b, b+n);
    _init(); 
  }
  
  void create();
  void optimize();
  void _init() { create(); optimize(); }

  ipairlist get_optimal_path() const;
  std::vector<int> get_optimal_path_extern();
  void write_dot() const;

private:
  int n_;
  int T_;
  int per_level_;
  fvec a_;
  fvec b_;
  ipairlist optimalpath_;
  ilist optimalnodepath_;
  ilist optimaledgeweights_;
  graph_t G_;

  inline int node_to_int(int,int);
  inline ipair int_to_node(int);
  float compute_weight(int, int);
  float compute_weight(int, int, fvec&);
  void add_edge_and_weight(int, int, fvec&&);
};

#endif
