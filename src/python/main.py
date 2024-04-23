from Graph import Graph
from paths import graphml_path, kp_paths, rpaths
from robustness import attack, verify_integrity
from visual import visualize_bc, visualize_edgeList, visualize_attack_scores, visualize_bc_distrs, visualize_Delta_bc
from procedures import extend_attack_procedure

import json
import networkx as nx
import matplotlib.pyplot as plt
import osmnx as ox
import numpy as np
from sys import setrecursionlimit

setrecursionlimit(100000)

def main():
    with open("data/nocost_1000_005.json", "r") as read_file:
        data = json.load(read_file)
    cut = data["8"]
    G = Graph(json=kp_paths[1])
    G.set_last_results(cut[0], cut[1])
    edges = G.process_cut()
    # attack on best cut types
    n = 10
    attack(G, n, "nocost_cut_bc_" + str(n) + ".json", "bc", True, True, subset=edges)
    
    # with open("data/robust/nocost_graph_bc_10.json", "r") as read_file:
    #     impt = json.load(read_file)
    # bc1 = {}
    # for k, v in impt[0][1].items():
    #     bc1[eval(k)] = v
    # bc2 = {}
    # for k, v in impt[1][1].items():
    #     bc2[eval(k)] = v
    # G_nx = ox.load_graphml(graphml_path[0])
    # visualize_Delta_bc(bc1, bc2, G_nx, "data/visubc_0-1_bc_rel.pdf", False, "rel bc evolution after removing 1 edge", treshold=None)
    
main()
