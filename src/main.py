from Graph import Graph
from paths import graphml_path, kp_paths, rpaths
from robustness import attack, extend_attack, cascading_failure
from visual import visualize_edgeList, visualize_bc, visualize_impact_evolution, visualize_impacts_comparison, visualize_bc_distrs
from procedures import efficiency_procedure, hundred_samples_eBC, bc_impact_procedure
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

def main():
    with open("data/cuts/lanes_1000_005.json", "r") as read_file:
        data = json.load(read_file)
    cut = data["24"] # 141
    G = Graph(json=kp_paths[9])
    G.set_last_results(cut[0], cut[1])
    edges = G.process_cut()
    G_nx = ox.load_graphml(graphml_path[2])
    weigths = nx.get_edge_attributes(G_nx, "weight")
    new_weights = {}
    for k, v in weigths.items():
        new_weights[k] = int(v)
    nx.set_edge_attributes(G_nx, new_weights, "weight")
    for i in range(0, 10):
        print(f"computing {i}th cut24")
        bc = nx.edge_betweenness_centrality(G_nx, weight="weight")
        if i > 0:
            print(edges)
            chosen_edge = edges.pop(rd.randint(0, len(edges)-1))
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
