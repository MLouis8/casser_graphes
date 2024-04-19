from Graph import Graph
from paths import graphml_path, kp_paths, rpaths
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
    with open("data/cuts/nocost_1000_005.json", "r") as read_file:
        data = json.load(read_file)
    cut = data["8"]
    G = Graph(json=kp_paths[1])
    G.set_last_results(cut[0], cut[1])
    edges = G.process_cut()
    # attack on best cut types
    n = 10
    attack(G, n, "data/robust/nocost_cut_bc_" + str(n) + ".json", "bc", True, True, subset=edges)
    
    # visualize
    # with open("data/ParisBClanes.json", "r") as read_file:
    #     impt = json.load(read_file)
    # bc = {}
    # for k, v in impt.items():
    #     bc[eval(k)] = v
    # G_nx = ox.load_graphml(graphml_path[2])
    # visualize_bc(bc, G_nx, "data/test.pdf", "bc")
    
    # extend attack
    # G = Graph(json=kp_paths[1])
    # extend_attack_procedure(rpaths[0], rpaths[1], G=G, k=5, fp_save="", order="bc", metric_bc=True, metric_cc=True)
main()
