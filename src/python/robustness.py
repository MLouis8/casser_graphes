from python.Graph import Graph
from python.typ import RobustnessDict, Cuts, Edge, EdgeDict

import json
import random as rd
import networkx as nx
import numpy as np
from typing import Any

def freq_attack(G: Graph, ncuts: int) -> Edge:
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
    return max(frequencies, key=frequencies.get)

def betweenness_attack(G: Graph):
    bc = G.get_edge_bc(new=True)
    return max(bc, key=bc.get)

def random_attack(G: Graph):
    return rd.choice(G.edges)

def attack(
    G: Graph, k: int, fp_save: str, order: str, metric_bc: bool, metric_cc: bool, ncuts: int = 1000
) -> RobustnessDict:
    """
    Simulates an attack on a Graph with the following strategy:
        repeat k times:
        - cuts ncuts times G
        - remove the most cut edge

    Parameters:
        G: Graph
        k: int, the number of edges to remove
        fp_save: str, the saving path
        order: str, can be "bc", "freq" or "rd" depending on the strategy to apply for the attack
            (ex. bc will remove the highest Betweenness Centrality edge)
        metric_bc: bool, whether the Betweenness Centrality is computed at each step
        metric_cc: bool, whether the max size of the connected components is computed or not
        ncuts: int (default 1000), the number of cuts to decide

    Saves:
        List of size k, containing a tuple of size 3 containing:
         - removed edge
         - 0, 1 or 2 metrics applied to the graph at this step
    """
    metrics, chosen_edge = [], None
    for i in range(k):
        bc = G.get_edge_bc() if metric_bc else None
        cc = G.get_max_cc() if metric_cc else None
        metrics.append((chosen_edge, bc, cc))
        match order:
            case "bc":
                chosen_edge = betweenness_attack(G)
            case "freq":
                chosen_edge = freq_attack(G, ncuts)
            case "rd":
                chosen_edge = random_attack(G)
        G.remove_edge(chosen_edge)
    bc = G.get_edge_bc() if metric_bc else None
    cc = G.get_biggest_connected_component() if metric_cc else None
    metrics.append((chosen_edge, bc, cc))

    with open(fp_save, "w") as save_file:
        json.dump(metrics, save_file)

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
    return avg1 / cpt1, avg2 / cpt2
