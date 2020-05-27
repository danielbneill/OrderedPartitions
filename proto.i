/* File : proto.i */
%module proto

%{
#include "graph.hpp"
#include "python_graph.hpp"
%}

%include "std_vector.i"

namespace std {
%template(IArray) vector<int>;
%template(FArray) vector<float>;
%template(IArrayArray) vector<vector<int>>;
}

%include "python_graph.hpp"

