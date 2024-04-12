from Graph import Graph

import json
import random as rd
from typing import Any
from typ import Cuts, Edge, RobustnessDict

def edge_frequency_attack(G: Graph, k: int, fp: str, ncuts: int=1000, bc: bool=False, cfbc: bool=False, dist: bool=False, spec_gap: bool=False, spec_rad: bool=False, nat_co: bool=False) -> RobustnessDict:
    """"
    Simulates an attack on a Graph with the following strategy:
        repeat k times:
        - cuts ncuts times G
        - remove the most cut edge

    Parameters:
        G: Graph
        k: int, the number of edges to remove
        ncuts: int (default 1000), the number of cuts to decide
    
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
        most_cut_edge = max()
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
        json.dump(robust_dict)


