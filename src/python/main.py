from Graph import Graph
from paths import graphml_path, kp_paths, rpaths
from robustness import attack, verify_integrity, extend_attack
from visual import visualize_bc, visualize_edgeList, visualize_attack_scores, visualize_bc_distrs, visualize_Delta_bc
from procedures import extend_attack_procedure, prepare_instance, init_city_graph, bc_difference_map_procedure
from geo import neighborhood_procedure

import json
import networkx as nx
import matplotlib.pyplot as plt
import osmnx as ox
import numpy as np
from sys import setrecursionlimit

setrecursionlimit(100000)

def main():
    # with open("data/lanes_1000_005.json", "r") as read_file:
    #     data = json.load(read_file)
    # cut = data["24"] # 141
    # G = Graph(json=kp_paths[1])
    # G.set_last_results(cut[0], cut[1])
    # edges = G.process_cut()
    # # attack on best cut types
    # n = 10
    # attack(G, n, "nocost_cut24_bc_" + str(n) + ".json", "bc", True, True, subset=edges)
    # with open("data/robust/nocost_graph_bc_10.json", "r") as read_file:
    #     metrics = json.load(read_file)
    # extend_attack(G, metrics, n, "nocost_graph_bc_" + str(n+10) + ".json", "bc", True, True, 1000, 100, True)

    bc_difference_map_procedure("data/robust/nocost_graph_freq_10.json", "presentations/others/visubc_0-10_freq.pdf", graphml_path[0])
    
main()
