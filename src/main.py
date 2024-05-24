from Graph import Graph
from paths import graphml_path, kp_paths, robust_paths_directed, effective_res_paths
from robustness import attack, cpt_effective_resistance
from visual import cumulative_impact_comparison, visualize_bc, visualize_city_parts, visualize_edgeList
from procedures import procedure_compare_scc, extend_attack, thousand_cuts, procedure_effective_resistance
from geo import neighborhood_procedure
from communities import louvain_communities_wrapper, walktrap_communities_wrapper
from cdlib.algorithms import walktrap

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
    G_nx: nx.Graph = ox.load_graphml(graphml_path[2])
    # G = Graph(json=kp_paths[9])
    # with open("data/cuts/lanes_1000_005.json", "r") as read_file:
    #     data = json.load(read_file)
    # cut = data['24'] # 141 or 190
    # G.set_last_results(cut[0], cut[1])
    # edges = G.process_cut()
    
    # with open("data/robust/weighted/lanes_graph_freq_45_new.json", "r") as robust_file:
    #     metrics = json.load(robust_file)
    # extend_attack(G, metrics, 3, "data/robust/weighted/lanes_graph_freq_48_new.json", "freq", True, True, True, 1000, True, weighted=True)#, subset=edges)
    
    with open("data/robust/directed/cut24dirbc_edges.json", 'r') as rfile:
        edges = json.load(rfile)
    # with open("data/robust/directed/cut24dirrd_edges.json", 'r') as rfile:
    #     edges = json.load(rfile)
    # with open("data/robust/directed/cut141dirbc_edges.json", 'r') as rfile:
    #     edges = json.load(rfile)
    # with open("data/robust/directed/cut141dirrd_edges.json", 'r') as rfile:
    #     edges = json.load(rfile)
    redges = [eval(edge) for edge in edges]
    procedure_effective_resistance(G_nx, redges, effective_res_paths[3], True)
    
    # with open('data/robust/directed/lanes_cut24dir_rd_40.json', 'r') as rfile:
    #     data = json.load(rfile)
    # with open('data/robust/directed/cut141dirbc_edges.json', 'r') as rfile:
    #     edges = json.load(rfile)
    # post_p = []
    # flag = True
    # for attack in data:            
    #     if isinstance(attack[0], list):
    #         post_p.append([str((eval(attack[0][0]), eval(attack[0][1])))] + attack[1:])
    #     elif not attack[0] or not eval(attack[0]):
    #         if flag:
    #             post_p.append(attack)
    #             flag = False
    #         else:
    #             continue
    #     else:
    #         post_p.append(attack)
    # print(len(post_p))
    # post_p = []
    # for i, attack in enumerate(data):
    #     post_p.append([edges[i], data[1]])
    # with open('data/robust/directed/lanes_cut141dir_bc_40p.json', 'w') as wfile:
    #     json.dump(post_p, wfile)

main()
