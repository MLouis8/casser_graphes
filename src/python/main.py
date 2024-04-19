from Graph import Graph
from paths import graphml_path, kp_paths
from robustness import attack
from visual import visualize_bc, visualize_edgeList
from procedures import extend_attack_procedure

import json
import networkx as nx
import matplotlib.pyplot as plt
import osmnx as ox
from sys import setrecursionlimit

setrecursionlimit(100000)

def main():
    # G = Graph(json=kp_paths[1])  # no cost
    # n = 5
    # attack(G, n, "data/nocost_graph_RD_" + str(n) + ".json", "rd", True, True)
    # with open("data/ParisBClanes.json", "r") as read_file:
    #     impt = json.load(read_file)
    # bc = {}
    # for k, v in impt.items():
    #     bc[eval(k)] = v
    # G_nx = ox.load_graphml(graphml_path[2])
    # visualize_bc(bc, G_nx, "data/test.pdf", "bc")
main()
