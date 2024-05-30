from Graph import Graph
from paths import graphml_path, kp_paths, dir_paths, effective_res_paths, redges_paths, eff_paths_dir
from robustness import attack, cpt_effective_resistance
from visual import cumulative_impact_comparison, visualize_bc, visualize_city_parts, visualize_edgeList, compare_avgebc_efficiency
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
    # G = Graph(json=kp_paths[9])#, nx=G_nx)
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
    # attack(G, 100, 'data/robust/bigattacks/freqrec_03_200.json', 'freq', False, False, False, ncuts=200, imb=0.3)
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
    
    with open(redges_paths[4], 'r') as rfile:
        redges = json.load(rfile)
    edges = [eval(redge) for redge in redges]
    for i in range(1, 40):
        try:
            G_nx.remove_edge(edges[i][0], edges[i][1])
        except:
            G_nx.remove_edge(edges[i][1], edges[i][0])
    procedure_effective_resistance(G_nx, edges[40:], effective_res_paths[10], True)

    # imbs = [0.05, 0.1, 0.2, 0.3]
    # nblocs = [2, 3]
    # ncuts = [1, 200, 1000]
    # for imb in imbs:
    #     for bloc in nblocs:
    #         for n in ncuts:
    #             start = time.time()
    #             for i in range(n):
    #                 G.kaffpa_cut(bloc, imb, 0, n, 2)
    #             stop = time.time()
    #             print(f"for imb={imb}, nbloc={bloc} and n={n}: time is {end-start}")
  
    # i = 9
    # procedure_global_efficiency(G_nx, redges_paths[i], eff_paths_dir[i], True)
  
    # compare_avgebc_efficiency(dir_paths, eff_paths_dir, ['eBC', 'freq', 'rd', 'deg', '24bc', '24rd', '141bc', '141rd', '190bc', '190rd'], "presentations/images/pres9/avgeBC_efficiencyALL.pdf")
    


main()
