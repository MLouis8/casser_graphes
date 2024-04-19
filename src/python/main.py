from Graph import Graph
from paths import graphml_path, kp_paths
from robustness import attack, extend_attack

import json
import networkx as nx
import matplotlib.pyplot as plt
from sys import setrecursionlimit

setrecursionlimit(100000)

def main():
    G = Graph(json=kp_paths[1]) # no cost
    n = 5
    attack(G, n, "data/nocost_graph_RD_"+str(n)+".json", "rd", True, True)
main()
