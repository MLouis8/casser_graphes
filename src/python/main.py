from Graph import Graph
from paths import graphml_path, kp_paths, rpaths
from robustness import attack, verify_integrity, extend_attack, measure_strong_connectivity, cpt_eBC_without_div, measure_bc_impact
from visual import visualize_bc, visualize_edgeList, visualize_attack_scores, visualize_bc_distrs, visualize_Delta_bc, visualize_edgeList_ordered
from procedures import extend_attack_procedure, prepare_instance, init_city_graph, bc_difference_map_procedure, thousand_cuts, clustering_procedure
from geo import neighborhood_procedure

from time import time

import json
import networkx as nx
import matplotlib.pyplot as plt
import osmnx as ox
import numpy as np
from sys import setrecursionlimit

setrecursionlimit(100000)

def main():
    G_nx = ox.load_graphml(graphml_path[2])
    # G = Graph(json="data/costs/lanes.json")
    # with open("data/cuts/lanes_1000_005.json", "r") as read_file:
    #     data = json.load(read_file)
    # cut = data["141"] # 24
    # G = Graph(json=kp_paths[9])
    # G.set_last_results(cut[0], cut[1])
    # edges = G.process_cut()
    # with open("data/robust/lanes_graph_deg_10_new.json", "r") as robust_file:
    #     metrics = json.load(robust_file)
    # attack(G, 10, "data/robust/lanes_graph_rd_10_new.json", "rd", True, True, nrandoms=1)
    # bcmodif = cpt_eBC_without_div(G_nx)
    # with open("data/robust/lanes_cut24_bc_25.json", "r") as freq_file:
    #     metrics = json.load(freq_file)
    # extend_attack(G, metrics, 40, "data/robust/lanes_graph_deg_50_new.json", "deg", True, True, 1000, 1, True, weighted=True)
    # with open("data/robust/lanes_graph_bc_25.json", "r") as read_file:
    #     data1 = json.load(read_file)
    # r_edge = eval(data1[1][0])
    # bc1 = {eval(k): v for k, v in data1[0][1].items()}
    # bc2 = {eval(k): v for k, v in data1[1][1].items()}
    # data = measure_bc_impact(bc1, bc2, r_edge, G_nx)
    # with open("data/robust/bcwithoutdiv_0.pdf", "w") as write_file:
    #     json.dump(bcmodif, write_file)
    # create visubcs
    # for i in range(11, 26):
    # i = 7
    # print(f"visualizing difference between {3} and {i}")
    # bc_difference_map_procedure(3, i, "data/robust/lanes_cut141_bc_10.json", "presentations/images/lanes/cut141/visubc_"+str(3)+"-"+str(i)+"_cut141.pdf", graphml_path[2], "bc", False)
    # with open("data/robust/lanes_cut141_bc_10.json", "r") as read:
    #     data = json.load(read)
    # bc = {}
    # for k, v in data[10][1].items():
    #     bc[eval(k)] = v
    # visualize_bc([(eval(a[0][0]), eval(a[0][1])) if a[0] else None for a in data], bc, G_nx, "data/visubc_10_cut141.pdf", "eBC after 10 removals for lane graph")#"base eBC for lane graph"
    
    
    # clustering_procedure(graphml_path[2], "data/costs/laneswithoutbridges200.json", "data/cuts/laneswithoutbridges200_1000_005.json", "laneswithoutbridges200", 7500)
    
    # with open("data/robust/lanes_graph_bc_50.json", "r") as robust_file:
    #     attacks = json.load(robust_file)

    # bc1 = attacks[0][1]
    # bc2 = attacks[1][1]
    # bc_diff = {}
    # for k, v in bc1.items():
    #     if k == '(4886, 4895)':
    #         continue
    #     try:
    #         if abs(v-bc2[k]) > 0.05:
    #             bc_diff[eval(k)] = abs(v-bc2[k])
    #     except:
    #         edge = (eval(k)[1], eval(k)[0])
    #         if abs(v-bc2[edge]) > 0.05:
    #             bc_diff[edge] = abs(v-bc2[edge])
    # nodes = set()
    # for edge in bc_diff.keys():
    #     nodes.add(edge[0])
    #     nodes.add(edge[1])
    
    # print("computing subset bc 1")
    # primebc1 = nx.edge_betweenness_centrality_subset(G_nx, nodes.union({4886, 4895}), nodes.union({4886, 4895}))
    # print("computing subset bc 2")
    # primebc2 = nx.edge_betweenness_centrality_subset(G_nx, nodes, nodes)
    # bc_diff2 = {}
    # for k, v in primebc1.items():
    #     if k == '(4886, 4895)':
    #         continue
    #     try:
    #         if abs(v-primebc2[k]) > 0.001:
    #             bc_diff2[k] = abs(v-primebc2[k])
    #     except:
    #         edge = (k[1], k[0])
    #         if abs(v-primebc2[edge]) > 0.001:
    #             bc_diff2[edge] = abs(v-primebc2[edge])
    # print(bc_diff)
    # print("then")
    # print(bc_diff2)

    start = time()
    nx.edge_betweenness_centrality(G_nx, weight=True)
    end = time()
    print("computing bc took ", end-start, "seconds")
main()
