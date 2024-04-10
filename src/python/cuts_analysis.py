from scipy.stats import pearsonr
import networkx as nx
import numpy as np
from Graph import Graph

from typing import Any
from typ import KCuts, EdgeDict, EdgeDictStr, Classes, Cuts, Edge

def determine_edge_frequency(G: Graph, C: dict[str, KCuts]) -> EdgeDict:
    """
    Function for determining edge frequency in cuts

    Paramters:
        G: Graph
        C: dictionary of cuts indexed by cut name: {'cut_name': (edgecut: int, blocks: list[int])}
    """
    frequencies: EdgeDict = {}

    for _, val in C.items():
        G.set_last_results(*val)
        p_cut = G.process_cut(weight=True)
        for edge in p_cut:
            if edge[:2] in frequencies:
                frequencies[edge[:2]] += 1
            else:
                frequencies[edge[:2]] = 1

    # removing doublons
    freq = frequencies.copy()
    for n1, n2 in freq:
        if (n2, n1) in frequencies.keys() and (n1, n2) in frequencies.keys():
            frequencies.pop((n2, n1))
    return frequencies


def cpt_cuts_correlation(edge_count: EdgeDict, measure: EdgeDictStr):
    """
    Analyses the correlation between the edges cut frequency and an arbitrary measure.

    Parameters:
        edge_count: dict(edge (tuple[int, int]): measure (float|int))
        measure: dict(edge str(tuple[int, int]): measure (float|int))

    Warning dicts keys must correspond.
    """
    x, y, nb_count = [], [], len(edge_count.keys())
    for edge in edge_count.keys():
        x.append(edge_count[edge] / nb_count)
        edge_converted = str((edge[0], edge[1], 0))
        edge_converted_r = str((edge[1], edge[0], 0))
        try:
            y.append(measure[edge_converted])
        except:
            y.append(measure[edge_converted_r])
    return pearsonr(x, y)


def get_n_biggest_freq(freq: EdgeDict, n: int) -> EdgeDict:
    """Return dict of edges: count of the n most frequent edges"""
    chosen = {}
    for k, v in freq.items():
        chosen[k] = v
        if len(chosen.keys()) > n:
            chosen.pop(min(chosen, key=chosen.get))
    return chosen


def to_Cut(xadj: list[int], adjncy: list[int], blocks: list[int]) -> list[tuple[int, int]]:
    edges = []
    for i in range(1, len(xadj)):
        for j in range(xadj[i - 1], xadj[i]):
            edges.append((i - 1, adjncy[j]))
    cut_edges = []
    for edge in edges:
        if blocks[edge[0]] != blocks[edge[1]]:
            if not edge in cut_edges and not (edge[1], edge[0]) in cut_edges:
                cut_edges.append(edge)
    return cut_edges

def get_connected_components(G_kp: Graph) -> Any:
    """Get connected components from KaHIP graph using NetworkX"""
    G_nx = G_kp.to_nx()
    cut = G_kp.process_cut()
    G_nx.remove_edges_from(cut)
    return nx.connected_components(G_nx)

def classify_by_connected_components(cc: dict[str, list[int]], liberty: int=3) -> Classes:
    classes: Classes = []
    for cut_name, co_cpnts in cc.items():
        classified = False
        for class_id, cls in enumerate(classes):
            if cc[cls[0]][:liberty] == co_cpnts[:liberty]:
                classified = True
                break
        if classified:
            classes[class_id].append(cut_name)
        else:
            classes.append([cut_name])
    return classes

def class_mean_cost(cls: list[str], cuts: Cuts, G: nx.Graph):
    weights = nx.get_edge_attributes(G, "weight")
    cost = []
    for cut_name in cls:
        for name, edges in cuts.items():
            if cut_name == name:
                for edge in edges:
                    cost.append(weights[(edge[0], edge[1], 0)])
    return np.mean(np.array(cost))
