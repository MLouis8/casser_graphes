from Graph import Graph
from paths import graphml_path, kp_paths, robust_paths_directed, effective_res_paths
from robustness import attack, cpt_effective_resistance
from visual import cumulative_impact_comparison, visualize_bc, visualize_city_parts
from procedures import procedure_compare_scc, extend_attack, thousand_cuts, procedure_effective_resistance
from geo import neighborhood_procedure
# from communities import louvain_communities_wrapper, walktrap_communities_wrapper

from time import time
import random as rd
import json
import networkx as nx
import matplotlib.pyplot as plt
import osmnx as ox
import numpy as np
from sys import setrecursionlimit

setrecursionlimit(100000)

def equal_dict(d1, d2) -> bool:
    if len(list(d1.keys())) != len(list(d2.keys())):
        print(len(list(d1.keys())), len(list(d2.keys())))
        return False
    for k, v in d1.items():
        if v != d2[k]:
            print(v, k, d2[k])
            return False
    return True

def main():
    G_nx = ox.load_graphml(graphml_path[2])
    # walktrap_communities_wrapper(G_nx, "data/walktrap_base.json", k = 4)
    # louvain_communities_wrapper(G_nx, "data/louvain_base.json", res=0.004)
    # with open("data/louvain_base.json", "r") as rfile:
    #     walktrap = json.load(rfile) 
    
    # with open("data/cuts/lanes_1000_005.json", "r") as read_file:
    #     data = json.load(read_file)
    # cut = data["24"] # 24
    G = Graph(json=kp_paths[9])
    # G.set_last_results(cut[0], cut[1])
    # edges = G.process_cut()
    # with open("data/robust/weighted/lanes_graph_freq_45_new.json", "r") as robust_file:
    #     metrics = json.load(robust_file)
    # extend_attack(G, metrics, 3, "data/robust/weighted/lanes_graph_freq_48_new.json", "freq", True, True, True, 1000, True, weighted=True)#, subset=edges)
    with open("data/robust/directed/bcdir_redges.json", 'r') as rfile:
        edges = json.load(rfile)
    redges = [eval(edge) for edge in edges]
    procedure_effective_resistance(G_nx, redges, effective_res_paths[0], True)
main()
