/* File : proto.i */
%module proto

%{
#include "graph.hpp"
#include "python_graph.hpp"
%}

%include "std_vector.i"
%include "std_pair.i"

namespace std {
%template(IArray) vector<int>;
%template(FArray) vector<float>;
%template(IArrayArray) vector<vector<int>>;
%template(IArrayFPair) pair<vector<vector<int>>, float>;
}

%include "python_graph.hpp"

