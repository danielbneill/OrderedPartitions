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
  EdgeWeightProperty(double w) : weight(w) {}
  double weight;
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
using ivec = std::vector<int>;
using fvec = std::vector<double>;
using ivecvec = std::vector<std::vector<int>>;

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
    per_level_{n-T+1},
    priority_sortind_{ivec(T_)},
    subsets_{ivecvec(T_)},
    optimalweight_{0.}
  { _init(); }

  PartitionGraph(int n,
		 int T,
		 double *a,
		 double *b
		 ):
    n_{n},
    T_{T},
    per_level_{n-T+1},
    priority_sortind_{ivec(T_)},
    subsets_{ivecvec(T_)},
    optimalweight_{0.}
  { 
    a_.assign(a, a+n);
    b_.assign(b, b+n);
    _init(); 
  }
  
  void create();
  void optimize();
  void _init() { create(); optimize(); }

  ipairlist get_optimal_path() const;
  ivec get_optimal_path_extern() const;
  ivecvec get_optimal_subsets_extern() const;
  double get_optimal_weight_extern() const;
  void write_dot();

private:
  int n_;
  int T_;
  int per_level_;
  fvec a_;
  fvec b_;
  ivec priority_sortind_;
  ipairlist optimalpath_;
  ilist optimalnodepath_;
  double optimalweight_;
  ilist optimaledgeweights_;
  ivecvec subsets_;
  graph_t G_;

  inline int node_to_int(int,int);
  inline ipair int_to_node(int);
  void sort_by_priority(fvec&, fvec&);
  double compute_weight(int, int);
  double compute_weight(int, int, fvec&);
  void add_edge_and_weight(int, int, fvec&&);
};

#endif
