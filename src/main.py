from Graph import Graph
from paths import graphml_path, kp_paths, dir_paths, effective_res_paths, redges_paths, eff_paths_dir, bigattacks_paths, bigattacks_paths2
from robustness import attack, cpt_effective_resistance
from visual import cumulative_impact_comparison, visualize_bc, visualize_Delta_bc, visualize_city_parts, visualize_edgeList, compare_avgebc_efficiency
from procedures import procedure_compare_scc, extend_attack, thousand_cuts, procedure_effective_resistance, clustering_procedure, verify_robust_list_integrity,procedure_global_efficiency
from geo import neighborhood_procedure
from communities import louvain_communities_wrapper, walktrap_communities_wrapper, determine_cut_edges
# from cdlib.algorithms import walktrap

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
    G_nx: nx.Graph = ox.load_graphml(graphml_path[2])
    # G_nx = G_nx.to_undirected()
    # w = nx.get_edge_attributes(G_nx, 'weight')
    # new_w = {}
    # for k, v in w.items():
    #     new_w[k] = eval(v)
    # nx.set_edge_attributes(G_nx, new_w, 'weight')
    G = Graph(json=kp_paths[9])#, nx=G_nx)
    # with open("data/cuts/lanes_1000_005.json", "r") as read_file:
    #     data = json.load(read_file)
    # cut = data['190'] # 141 or 24
    # G.set_last_results(cut[0], cut[1])
    # edges = G.process_cut()

    # attack(G, 100, 'data/robust/bigattacks/bcapprox_100.json', 'bc', False, False, True, bc_approx=1000)
    # attack(G, 100, 'data/robust/bigattacks/freqrec_005_1000.json', 'freq', False, False, False, ncuts=1000)
    # attack(G, 100, 'data/robust/bigattacks/freqrec_01_1000.json', 'freq', False, False, False, ncuts=1000, imb=0.1)
    # attack(G, 100, 'data/robust/bigattacks/freqrec_02_1000.json', 'freq', False, False, False, ncuts=1000, imb=0.2)
    # attack(G, 100, 'data/robust/bigattacks/freqrec_03_1000.json', 'freq', False, False, False, ncuts=1000, imb=0.3)
    # attack(G, 100, 'data/robust/bigattacks/freqreck3_005_1000.json', 'freq', False, False, False, ncuts=1000, nblocks=3)
    # attack(G, 100, 'data/robust/bigattacks/freqrec_005_200.json', 'freq', False, False, False, ncuts=200)
    # attack(G, 100, 'data/robust/bigattacks/freqrec_01_200.json', 'freq', False, False, False, ncuts=200, imb=0.1)
    # attack(G, 100, 'data/robust/bigattacks/freqrec_02_200.json', 'freq', False, False, False, ncuts=200, imb=0.2)
    # attack(G, 100, 'data/robust/bigattacks/freqrec_k3_005_10.json', 'freq', False, False, False, ncuts=10, imb=0.05, nblocks=3)
    # attack(G, 100, 'data/robust/bigattacks/freq_rec005_k3.json', 'freq', False, False, False, ncuts=200, nblocks=3)

    # with open("data/robust/weighted/lanes_graph_freq_45_new.json", "r") as robust_file:
    #     metrics = json.load(robust_file)
    # extend_attack(G, metrics, 3, "data/robust/weighted/lanes_graph_freq_48_new.json", "freq", True, True, True, 1000, True, weighted=True)#, subset=edges)
    # 7
    # with open(dir_paths[7], 'r') as rfile:
    #     robust_list = json.load(rfile)
    # print(len(robust_list))
    # for attack in robust_list:
    #     print(attack[0])
    # verify_robust_list_integrity(7)
    
    # with open(redges_paths[1], 'r') as rfile: # 8 and 9
    #     redges = json.load(rfile)
    # edges = [eval(redge) for redge in redges]
    # for i in range(1, 14):
    #     try:
    #         G_nx.remove_edge(edges[i][0], edges[i][1])
    #     except:
    #         G_nx.remove_edge(edges[i][1], edges[i][0])
    # procedure_effective_resistance(G_nx, edges[14:], effective_res_paths[1], True)
    
    # G_nx = G.to_nx(directed=True)
    # print(G_nx.graph["crs"])
    # i = 4
    # with open(dir_paths[i], 'r') as rfile:
    #     data = json.load(rfile)
    # with open(redges_paths[i], 'r') as rfile:
    #     edges = json.load(rfile)
    # bc_dict1 = {eval(k): v for k, v in data[0][1].items()}
    # bc_dict2 = {eval(k): v for k, v in data[40][1].items()}
    # redges = [eval(edge) for edge in edges]
    # visualize_Delta_bc(redges, bc_dict1, bc_dict2, G_nx, 'presentations/images/directed_triples/triplevisucut190rddir2.pdf', False, 'eBC relative differences')
    # visualize_bc(redges, bc_dict2, G_nx, 'data/cut24test.pdf', 'eBC values')
    procedure_compare_scc(G_nx, bigattacks_paths, ['bcapprox', 'freq imb=0.05', 'freq imb=0.1', 'freq imb=0.2', 'freq imb=0.3', 'freq nblocks=3 & imb=0.05'], 'presentations/images/pres9/bigattacks_scc3.pdf')
    # procedure_compare_scc(G_nx, bigattacks_paths2, ['ncuts=1000', 'ncuts=200', 'ncuts=100', 'ncuts=10'], 'presentations/images/pres9/bigattacks_freq_approx_03.pdf')
main()
