from Graph import Graph
from paths import graphml_path, kp_paths, rpaths
from robustness import attack, verify_integrity, extend_attack, measure_strong_connectivity
from visual import visualize_bc, visualize_edgeList, visualize_attack_scores, visualize_bc_distrs, visualize_Delta_bc, visualize_edgeList_ordered
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
    G_nx = ox.load_graphml(graphml_path[2])

    with open("data/cuts/lanes_1000_005.json", "r") as read_file:
        data = json.load(read_file)
    # cut = data["141"] # 24
    # G = Graph(json=kp_paths[9])
    # G.set_last_results(cut[0], cut[1])
    # edges = G.process_cut()
    # attack on best cut types
    n = 25
    # attack(G, n, "lanes_graph_rd_" + str(n) + ".json", "rd", True, True, nrandoms=1)
    with open("data/robust/lanes_graph_deg_25.json", "r") as read_file:
        metrics = json.load(read_file)
    extend_attack(G_nx, metrics, n, "lanes_graph_deg_" + str(n+25) + ".json", "deg", True, True, 1000, 100, True)
    
    # create visubcs
    # for i in range(11, 26):
    # i = 7
    # print(f"visualizing difference between {3} and {i}")
    # bc_difference_map_procedure(3, i, "data/robust/lanes_cut141_bc_10.json", "presentations/images/lanes/cut141/visubc_"+str(3)+"-"+str(i)+"_cut141.pdf", graphml_path[2], "bc", False)
    # with open("data/robust/lanes_cut24_bc_10.json", "r") as read:
    #     data = json.load(read)
    # bc = {}
    # for k, v in data[0][1].items():
    #     bc[eval(k)] = v
    # visualize_bc([], bc, G_nx, "data/visubc_0.pdf", "Basic eBC for lane graph")
    
main()
