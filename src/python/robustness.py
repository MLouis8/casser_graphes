from Graph import Graph

import json
import random as rd
import networkx as nx
import numpy as np
from typing import Any
from typ import RobustnessDict, Cuts, Edge

def edge_frequency_attack(G: Graph, k: int, fp: str, ncuts: int=1000, bc: bool=False, cfbc: bool=False, dist: bool=False, spec_gap: bool=False, spec_rad: bool=False, nat_co: bool=False) -> RobustnessDict:
    """"
    Simulates an attack on a Graph with the following strategy:
        repeat k times:
        - cuts ncuts times G
        - remove the most cut edge

    Parameters:
        G: Graph
        fp: str, the saving filepath
        k: int, the number of edges to remove
        ncuts: int (default 1000), the number of cuts to decide
        bc: bool, whether Betweenness Centrality should be computed
        cfbc: bool, whether Current Flow Betweenness Centrality should be computed
        dist: bool, whether Distance should be computed
        spec_gap: bool, whether Spectral Gap should be computed
        spec_rad: bool, whether Spectral Radius should be computed
        nat_co: bool, whether Natural Connectivity should be computed

    Returns:
        A Robustness dictionary (see typ for more info)
    """
    print("computing basic robustness values")
    robust_dict = {
        "edges": [],
    }
    if bc:
        robust_dict["avg bc"] = [G.get_avg_edge_bc]
    if cfbc:
        robust_dict["avg cf bc"] = [G.get_avg_edge_cf_bc]
    if dist:
        robust_dict["avg dist"] = [G.get_avg_dist]
    if spec_gap:
        robust_dict["spectral gap"] = [G.get_spectral_gap]
    if spec_rad:
        robust_dict["spectral rad"] = [G.get_spectral_radius]
    for i in range(k):
        cut_union = []
        seen_seeds = []
        for ncut in range(ncuts):
            seed = rd.randint(0, 1044642763)
            while seed in seen_seeds:
                seed = rd.randint(0, 1044642763)
            seen_seeds.append(seed)
            G.kaffpa_cut(2, 0.05, 0, seed, 3)
            cut_union += G.process_cut()
        
        frequencies = {}
        for edge in cut_union:
            if edge in frequencies:
                frequencies[edge] += 1
            else:
                frequencies[edge] = 1
        most_cut_edge = max(frequencies, key=frequencies.get)
        G.remove_edge(most_cut_edge)
        robust_dict["edges"].append(most_cut_edge)
        if bc:
            robust_dict["avg bc"].append(G.get_avg_edge_bc)
        if cfbc:
            robust_dict["avg cf bc"].append(G.get_avg_edge_cf_bc)
        if dist:
            robust_dict["avg dist"].append(G.get_avg_dist)
        if spec_gap:
            robust_dict["spectral gap"].append(G.get_spectral_gap)
        if spec_rad:
            robust_dict["spectral rad"].append(G.get_spectral_radius)

    with open(fp, "w") as save_file:
        json.dump(robust_dict, save_file)
# Sur le modèle ci-dessus
# faire une random edge attack
# faire une random edge parmis celles coupées


def betweenness_attack(G: nx.Graph, k: int) -> RobustnessDict:
    robust_dict = {
        "edges": [],
        "avg bc": []
    }

    for _ in range(k):
        bc_dict = nx.edge_betweenness_centrality(G)
        biggest_bc_edge = max(bc_dict, key=bc_dict.get)
        robust_dict["edges"].append(biggest_bc_edge)
        robust_dict["avg bc"].append(np.mean(list(bc_dict.values())))
        G.remove_edge(biggest_bc_edge[0], biggest_bc_edge[1])

    robust_dict["avg bc"].append(np.mean(list(nx.edge_betweenness_centrality(G).values())))
    return robust_dict

def best_class_attack(classes: list[Cuts], metric: str="bc"):
    """
    Test every class cut on the Graph and returns the best one
    """
    pass

def cpt_robust_dict_from_edges(G: Graph, edges: list[Edge], fp: str, bc: bool = False, cfbc: bool = False, dist: bool = False, spec_gap: bool = False, spec_rad: bool = False, nat_co: bool = False):
    """
    Takes as input a list of edges to remove one by one, and computes at each step the different robustness metrics

    Parameters:
        edges: list of edges to remove in order
        fp: str, filepath for saving
        bc: bool, whether Betweenness Centrality should be computed
        cfbc: bool, whether Current Flow Betweenness Centrality should be computed
        dist: bool, whether Distance should be computed
        spec_gap: bool, whether Spectral Gap should be computed
        spec_rad: bool, whether Spectral Radius should be computed
        nat_co: bool, whether Natural Connectivity should be computed
    """
    print("computing basic robustness values")
    robust_dict = {
        "edges": edges,
    }
    if bc:
        robust_dict["avg bc"] = [G.get_avg_edge_bc]
    if cfbc:
        robust_dict["avg cf bc"] = [G.get_avg_edge_cf_bc]
    if dist:
        robust_dict["avg dist"] = [G.get_avg_dist]
    if spec_gap:
        robust_dict["spectral gap"] = [G.get_spectral_gap]
    if spec_rad:
        robust_dict["spectral rad"] = [G.get_spectral_radius]
    for edge in edges:
        G.remove_edge(edge)
        if bc:
            robust_dict["avg bc"].append(G.get_avg_edge_bc)
        if cfbc:
            robust_dict["avg cf bc"].append(G.get_avg_edge_cf_bc)
        if dist:
            robust_dict["avg dist"].append(G.get_avg_dist)
        if spec_gap:
            robust_dict["spectral gap"].append(G.get_spectral_gap)
        if spec_rad:
            robust_dict["spectral rad"].append(G.get_spectral_radius)

    with open(fp, "w") as save_file:
        json.dump(robust_dict, save_file)

def avg_bc_edge_subset(G: nx.Graph, s: list[Edge]):
    """
    Returns the average BC of the all graph compared to the average of the listed edges,
    the graph must have the attribute betweenness
    """
    avg1, cpt1 = 0, 0
    avg2, cpt2 = 0, 0
    for edge in G.edges(data=True):
        avg1 += edge[2]["betweenness"]
        cpt1 += 1
        if (edge[0], edge[1]) in s or (edge[1], edge[0]) in s:
            avg2 += edge[2]["betweenness"]
            cpt2 += 1
    return avg1/cpt1, avg2/cpt2