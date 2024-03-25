#pragma once

#include <array>

using namespace std;

template <size_t V, size_t E>
class Graph
{
    array<int, V + 1> xadj;
    array<int, 2 * E> adjncy;
    array<int, V> vwgt;
    array<int, 2 * E> adjcwgt;

public:
    Graph(array<int, V + 1> xadj, array<int, 2 * E> adjncy, array<int, V> vwgt, array<int, 2 * E> adjcwgt) : xadj(xadj), adjncy(adjncy), vwgt(vwgt), adjcwgt(adjcwgt);

    size_t getSizeV() const { return V; }

    size_t getSizeE() const { return E; }

};