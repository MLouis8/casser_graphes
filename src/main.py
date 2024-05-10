from Graph import Graph
from paths import graphml_path, kp_paths, rpaths
from robustness import attack
from visual import visualize_edgeList, visualize_bc
from procedures import efficiency_procedure
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
    # G = Graph(json="data/costs/lanes.json")
    # with open("data/cuts/lanes_1000_005.json", "r") as read_file:
    #     data = json.load(read_file)
    # cut = data["141"] # 24
    G = Graph(json=kp_paths[9])
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

    # with open("data/robust/lanes_graph_bc_10_new.json", "r") as read:
    #     data = json.load(read)
    # bc = {}
    # for k, v in data[0][1].items():
    #     bc[eval(k)] = v
    # visualize_bc([], bc, G_nx, "data/visubclanes_0new.pdf", "basic Paris eBC lanes weighted")
    
    # clustering_procedure(graphml_path[2], "data/costs/laneswithoutbridges200.json", "data/cuts/laneswithoutbridges200_1000_005.json", "laneswithoutbridges200", 2000)
    # clustering_procedure(graphml_path[2], "data/costs/laneswithoutbridges200.json", "data/cuts/laneswithoutbridges200_1000_005.json", "laneswithoutbridges200", 3000)
    # clustering_procedure(graphml_path[2], "data/costs/laneswithoutbridges200.json", "data/cuts/laneswithoutbridges200_1000_005.json", "laneswithoutbridges200", 4000)
    
    # with open("data/robust/lanes_graph_bc_50.json", "r") as robust_file:
    #     attacks = json.load(robust_file)
    # efficiency_procedure(G_nx, "data/robust/lanes_graph_deg_50.json", "data/robust/lanesgraphdeg_efficiency_50.json")
    weigths = nx.get_edge_attributes(G_nx, "weight")
    new_weights = {}
    for k, v in weigths.items():
        new_weights[k] = int(v)
    nx.set_edge_attributes(G_nx, new_weights, "weight")

    for i in range(0, 51):
        print(f"computing {i}th bc")
        bc = nx.edge_betweenness_centrality(G_nx, weight="weight")
        if i > 0:
            edge = max(bc, key=bc.get)
            G_nx.remove_edge(edge[0], edge[1])
        else:
            edge = None
        with open("data/robust/lanes_graphdir_bc_50.json", "r") as rfile:
            data = json.load(rfile)
        data.append((edge, bc))
        with open("data/robust/lanes_graphdir_bc_50.json", "w") as wfile:
            json.dump(data, wfile)
main()
