from Graph import Graph
from paths import graphml_path, kp_paths, rpaths
from robustness import attack, extend_attack, cascading_failure
from visual import visualize_edgeList, visualize_bc, visualize_impact_evolution, visualize_impacts_comparison
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
    G_nx = ox.load_graphml(graphml_path[2])
    weigths = nx.get_edge_attributes(G_nx, "weight")
    new_weights = {}
    for k, v in weigths.items():
        new_weights[k] = int(v)
    nx.set_edge_attributes(G_nx, new_weights, "weight")
    with open("data/robust/weighted/lanes_graph_bc_10_new.json", "r") as rfile:
        data = json.load(rfile)
    maxbc = max(data[0][1].values())
    results = cascading_failure(G_nx, [data[1][0]], maxbc, (0, None), data[1][1])
    with open("data/test_cascading_bc.json", "w") as wfile:
        json.dump(results, wfile)
main()
