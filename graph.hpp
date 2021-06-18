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

#define MON_Internal_UnusedStringify(macro_arg_string_literal) #macro_arg_string_literal
#define MONUnusedParameter(macro_arg_parameter) _Pragma(MON_Internal_UnusedStringify(unused(macro_arg_parameter)))

//
// boost adjacency_list type
// adjacency_list<OutEdgeList, VertexList, Directed,
//                VertexProperties, EdgeProperties,
//                GraphProperites, EdgeList>
//
             
// typedef boost::property<boost::edge_weight_t, float> EdgeWeightProperty;

struct EdgeWeightProperty {
  EdgeWeightProperty() : weight(0) {}
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

class PartitionGraph {
public:
  PartitionGraph(int n, 
		 int T,
		 std::vector<float> a,
		 std::vector<float> b
		 ) :
    n_{n},
    T_{T},
    a_{a},
    b_{b},
    per_level_{n-T+1},
    priority_sortind_{std::vector<int>(T_)},
    optimalweight_{0.},
    subsets_{std::vector<std::vector<int>>(T_)}
  { _init(); }

  PartitionGraph(int n,
		 int T,
		 float *a,
		 float *b
		 ):
    n_{n},
    T_{T},
    per_level_{n-T+1},
    priority_sortind_{std::vector<int>(T_)},
    optimalweight_{0.},
    subsets_{std::vector<std::vector<int>>(T_)}
  { 
    a_.assign(a, a+n);
    b_.assign(b, b+n);
    _init(); 
  }
  
  void create();
  void optimize();
  void _init() { create(); optimize(); }

  std::list<std::pair<int, int>> get_optimal_path() const;
  std::vector<int> get_optimal_path_extern() const;
  std::vector<std::vector<int>> get_optimal_subsets_extern() const;
  float get_optimal_weight_extern() const;
  void write_dot();

private:
  int n_;
  int T_;
  std::vector<float> a_;
  std::vector<float> b_;
  int per_level_;
  std::vector<int> priority_sortind_;
  std::list<std::pair<int, int>> optimalpath_;
  std::list<int> optimalnodepath_;
  float optimalweight_;
  std::list<int> optimaledgeweights_;
  std::vector<std::vector<int>> subsets_;
  graph_t G_;

  inline int node_to_int(int,int);
  inline std::pair<int, int> int_to_node(int);
  void sort_by_priority(std::vector<float>&, std::vector<float>&);
  float compute_weight(int, int);
  float compute_weight(int, int, std::vector<float>&);
  void add_edge_and_weight(int, int, std::vector<float>&&);
};

#endif
