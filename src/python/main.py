from Graph import Graph
from paths import graphml_path, kp_paths, rpaths
from robustness import attack, verify_integrity, extend_attack
from visual import visualize_bc, visualize_edgeList, visualize_attack_scores, visualize_bc_distrs, visualize_Delta_bc
from procedures import extend_attack_procedure, prepare_instance

import json
import networkx as nx
import matplotlib.pyplot as plt
import osmnx as ox
import numpy as np
from sys import setrecursionlimit

setrecursionlimit(100000)

def main():
    with open("data/lanes_1000_005.json", "r") as read_file:
        data = json.load(read_file)
    cut = data["24"] # 141
    G = Graph(json=kp_paths[1])
    G.set_last_results(cut[0], cut[1])
    edges = G.process_cut()
    # attack on best cut types
    n = 10
    attack(G, n, "nocost_cut24_bc_" + str(n) + ".json", "bc", True, True, subset=edges)
    # with open("data/robust/nocost_graph_bc_10.json", "r") as read_file:
    #     metrics = json.load(read_file)
    # extend_attack(G, metrics, n, "nocost_graph_bc_" + str(n+10) + ".json", "bc", True, True, 1000, 100, True)

    # with open("data/robust/nocost_graph_bc_20.json", "r") as read_file:
    #     impt = json.load(read_file)
    # bc1 = {}
    # for k, v in impt[0][1].items():
    #     bc1[eval(k)] = v
    # bc2 = {}
    # for k, v in impt[20][1].items():
    #     bc2[eval(k)] = v
    # G_nx = ox.load_graphml(graphml_path[0])
    # visualize_bc_distrs(bc1, bc2, "data/distrbc_0-20_bc.pdf", ["bc -0 edge", "bc -20 edges"])
    
    # prepare_instance("data/ParisPreprocessedL.graphml", "data/costs/laneswithoutzbridge2.json", "lanes without bridge")
    # G = Graph(json="data/costs/laneswithoutzbridge2.json")
    # G_nx = G.to_nx()
    # weights = nx.get_edge_attributes(G_nx, "weight")
    # edge_l = []
    # for edge, w in weights.items():
    #     if w == 95099713:
    #         edge_l.append(edge)
    # G_nx2 = ox.load_graphml("data/ParisPreprocessedL.graphml")
    # visualize_edgeList(edge_l, G_nx2, filepath="data/testbridge.pdf")
main()
