from Graph import Graph
from paths import graphml_path, kp_paths, rpaths
from robustness import attack, verify_integrity
from visual import visualize_bc, visualize_edgeList, visualize_attack_scores
from procedures import extend_attack_procedure

import json
import networkx as nx
import matplotlib.pyplot as plt
import osmnx as ox
import numpy as np
from sys import setrecursionlimit

setrecursionlimit(100000)

def main():
    # with open("data/cuts/nocost_1000_005.json", "r") as read_file:
    #     data = json.load(read_file)
    # cut = data["8"]
    # G = Graph(json=kp_paths[1])
    # G.set_last_results(cut[0], cut[1])
    # edges = G.process_cut()
    # attack on best cut types
    # n = 10
    # attack(G, n, "nocost_graph_freq_" + str(n) + ".json", "freq", True, True)
    
    # visualize
    # with open("data/robust/nocost_graph_bc.json", "r") as read_file:
    #     impt = json.load(read_file)
    # bc = {}
    # for k, v in impt.items():
    #     bc[eval(k)] = v
    # G_nx = ox.load_graphml(graphml_path[2])
    # visualize_bc(bc, G_nx, "presentations/images/visubc.pdf", "bc")

    # correlation BC and osmid
    # G_nx = ox.load_graphml(graphml_path[0])
    # for edge in G_nx.edges(data=True):
    #     print(edge[2])
    with open("data/robust/nocost_graph_deg_10.json", "r") as deg_file:
        deg = json.load(deg_file)

    with open("data/robust/nocost_graph_bc_10.json", "r") as bc_file:
        bc_graph = json.load(bc_file)

    deg_attack = [attack[2] for attack in deg]
    bc_graph_attack = [attack[2] for attack in bc_graph]
    visualize_attack_scores([deg_attack, bc_graph_attack], ["max degree", "max bc on graph"], "data/biggest_cc.pdf", False, "biggest components size evolution: BC vs Deg")
main()
