from Graph import Graph
from paths import graphml_path, kp_paths, rpaths
from robustness import attack, extend_attack, cascading_failure
from visual import visualize_edgeList, visualize_bc, visualize_impact_evolution, visualize_impacts_comparison, visualize_bc_distrs, visualize_attack_scores, visualize_biggest_scc, visualize_Delta_bc
from procedures import efficiency_procedure, hundred_samples_eBC, bc_impact_procedure, bc_difference_map_procedure
from geo import neighborhood_procedure

from time import time
import random as rd
import json
import networkx as nx
import matplotlib.pyplot as plt
import osmnx as ox
import numpy as np
from sys import setrecursionlimit

setrecursionlimit(100000)


#     G_nx = G_nx.to_undirected()
#     for attack in degdata:
#         try:
#             edge = eval(attack[0])
#             G_nx.remove_edge(edge[0], edge[1])
#         except:
#             print(attack[0])
#         with open("data/robust/other/efficiency/globaleff_notweigtheddeg.json", "r") as file:
#             globeff = json.load(file)
#         globeff.append(nx.global_efficiency(G_nx))
#         with open("data/robust/other/efficiency/globaleff_notweigtheddeg.json", "w") as file:
#             json.dump(globeff, file)   

def main():
    with open("data/cuts/lanes_1000_005.json", "r") as read_file:
        data = json.load(read_file)
    cut = data["24"] # 141
    G1 = Graph(json=kp_paths[9])
    G1.set_last_results(cut[0], cut[1])
    edges = G1.process_cut()
    G_nx = Graph(json=kp_paths[9]).to_nx()
    weigths = nx.get_edge_attributes(G_nx, "weight")
    new_weights = {}
    for k, v in weigths.items():
        new_weights[k] = int(v)
    nx.set_edge_attributes(G_nx, new_weights, "weight")
    for i in range(0, 40):
        print(f"computing {i}th cut24")
        bc = nx.edge_betweenness_centrality(G_nx, weight="weight")
        print("bc computed")
        if i > 0:
            chosen_edge, max_bc = None, 0
            for edge in edges:
                if edge in bc and bc[edge] > max_bc:
                    max_bc = bc[edge]
                    chosen_edge = edge
            edges.remove(chosen_edge)
            # chosen_edge = edges.pop(rd.randint(0, len(edges)-1))
            G_nx.remove_edge(chosen_edge[0], chosen_edge[1])
        else:
            edge = None
        with open("data/robust/lanes_cut24dir_bc_max.json", "r") as rfile:
            data = json.load(rfile)
        bcsave = {}
        for k, v in bc.items():
            bcsave[str(k)] = v
        data.append((str(edge), bcsave))
        with open("data/robust/lanes_cut24dir_bc_max.json", "w") as wfile:
            json.dump(data, wfile)
main()
