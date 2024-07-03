from Graph import Graph
from paths import graphml_path, kp_paths, eff_paths_dir, bigattacks_paths, effective_res_paths, robust_paths_impacts
from robustness import attack, cpt_effective_resistance, cluster_attack, measure_bc_impact_cumulative
from visual import cumulative_impact_comparison, visualize_edgeList_ordered, visualize_city_parts, visualize_edgeList, visualize_cluster, impact_scatter
from procedures import compare_scc_or_cc_procedure, effective_resistance_procedure, clustering_procedure, cluster_data_procedure, preprocess_robust_import
from geo import neighborhood_procedure
from BIRCHClustering import CFTree

from time import time
import random as rd
import json
import networkx as nx
import matplotlib.pyplot as plt
import osmnx as ox
import numpy as np
from sys import setrecursionlimit

setrecursionlimit(100000)

def to_hex(c):
    f = lambda x: int(255 * x)
    return "#{0:02x}{1:02x}{2:02x}".format(f(c[0]), f(c[1]), f(c[2]))

def main():
    # G_nx: nx.Graph = ox.load_graphml(graphml_path[2])
    weights = nx.get_edge_attributes(G_nx, "lenght")
    new_weights = {}
    for k, v in weights.items():
        new_weights[k] = int(v)
    nx.set_edge_attributes(G_nx, new_weights, 'weight')
    G_nx = G_nx.to_undirected()
    G = Graph(json=kp_paths[9], nx=G_nx)

    attack(G, 500, 'data/robust/bigattacks10-1000/eBCA500.json', 'bc', False, True, bc_approx=1000)
    # attack(G, 200, 'data/robust/bigattacks10-1000/eBCA200_approx=4000.json', 'bc', False, True, bc_approx=4000)
    # code for lcc-imb
    # paths_ids = [9, 4, 5, 6]
    # labels = ['Fa (k=2 ε=0.05)', 'Fa (k=2 ε=0.1)', 'Fa (k=2 ε=0.2)', 'Fa (k=2 ε=0.3)']
    # linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
    # fig, ax = plt.subplots()
    # for i in range(4):
    #     with open(bigattacks_paths[paths_ids[i]], 'r') as rfile:
    #         bigattack = json.load(rfile)
    #     redges = [big[0] for big in bigattack[1:]]
    #     costs = []
    #     for e in redges:
    #         if type(e) == str:
    #             e = eval(e)
    #         try:
    #             costs.append(eval(weights[(e[0], e[1], 0)]))
    #         except:
    #             costs.append(eval(weights[(e[1], e[0], 0)]))
    #     cumul_cost = [sum(costs[:i+1]) for i in range(len(costs))]
    #     lcc = [big[1] for big in bigattack[1:]]
    #     ax.plot(cumul_cost, np.array(lcc) / len(G_nx.nodes), label=labels[i], linestyle=linestyles[i])        
    # ax.legend()
    # ax.set_xlabel('attack cumulative cost')
    # ax.set_ylabel('lcc size')
    # fig.savefig('data/article-images/lcc-imb-cost.pdf')

    # paths = [
    #     'data/robust/other/impacts/lanesgraphbc_NW_impacts(e-3).json',
    #     'data/robust/other/impacts/lanesgraphfreq_NW_impacts(e-3).json',
    #     'data/robust/other/impacts/lanesgraphdeg_NW_impacts(e-3).json'
    # ]
    # names = ['eBCA', 'CFA', 'DLA']
    # linestyles = ['solid', 'dashdot', 'dotted']
    # markers = ['.', '*', 's']
    # cumulative_impact_comparison(paths, 'sumdiffs', 'cumulative sum of eBC changes', names, linestyles, 'data/article-images/sumdiffs.pdf')
    # impact_scatter(paths, names, markers, 'data/article-images/scatter-plot.pdf')
main()
