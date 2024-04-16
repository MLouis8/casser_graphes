from python.Graph import Graph
from python.paths import graphml_path, kp_paths

import json
import networkx as nx
import matplotlib.pyplot as plt
from sys import setrecursionlimit

setrecursionlimit(100000)

def main():
    G_bc = Graph(json=kp_paths[9]).to_nx()
    with open("data/robustness/ParisBCLanes.json", "w") as save:
        json.dump(nx.edge_betweenness_centrality(G_bc, weight="weight"), save)
main()
