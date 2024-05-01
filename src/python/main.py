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

    # with open("data/cuts/lanes_1000_005.json", "r") as read_file:
    #     data = json.load(read_file)
    # cut = data["141"] # 24
    # G = Graph(json=kp_paths[9])
    # G.set_last_results(cut[0], cut[1])
    # edges = G.process_cut()
    # attack on best cut types
    # n = 25
    # attack(G, n, "lanes_graph_rd_" + str(n) + ".json", "rd", True, True, nrandoms=1)
    # with open("data/robust/lanes_graph_deg_10.json", "r") as read_file:
    #     metrics = json.load(read_file)
    # extend_attack(G, metrics, n, "lanes_graph_deg_" + str(n+10) + ".json", "deg", True, True, 1000, 100, True)
    
    # create visubcs
    # for i in range(1, 11):
    # # i = 10
    #     print(f"visualizing difference between {i-1} and {i}")
    #     bc_difference_map_procedure(i-1, i, "data/robust/nocost_cut141_bc_10.json", "presentations/images/cuts_nocost?/visubc_"+str(i-1)+"-"+str(i)+"_cut141.pdf", graphml_path[0], "bc", False)
    with open("data/robust/lanes_cut24_bc_10.json", "r") as read:
        data = json.load(read)
    bc = {}
    for k, v in data[0][1].items():
        bc[eval(k)] = v
    visualize_bc([], bc, G_nx, "data/visubc_0.pdf", "Basic eBC for lane graph")
    # with open("data/robust/nocost_cut_bc_10.json", "r") as read:
    #     data = json.load(read)
    # with open("data/robust/nocost_graph_freq_10.json", "r") as read:
        # data2 = json.load(read)
    # bc1 = data[10][1]
    # bc2 = data2[10][1]
    # visualize_bc_distrs(bc1, bc2, "data/nocostcut_bcdistr_cut-freqbc.pdf", ("cut then bc", "freq"))

    # with open("data/robust/nocost_graph_bc_10.json", "r") as read:
    #     data3 = json.load(read)
    
    # attack1 = [a[1] for a in data]
    # attack2 = [a[1] for a in data2]
    # attack3 = [a[1] for a in data3]
    # visualize_attack_scores([attack1, attack2, attack3], ["cut", "freq", "bc"], "data/avgbc_cutvsfreq_nocost.pdf", True, "Average eBC evolution for 3 different attacks")

main()
