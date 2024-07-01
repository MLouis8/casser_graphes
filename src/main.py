from Graph import Graph
from paths import graphml_path, kp_paths, eff_paths_dir, bigattacks_paths, effective_res_paths
from robustness import attack, cpt_effective_resistance, cluster_attack
from visual import cumulative_impact_comparison, visualize_edgeList_ordered, visualize_city_parts, visualize_edgeList, visualize_cluster
from procedures import compare_scc_or_cc_procedure, effective_resistance_procedure, clustering_procedure, cluster_data_procedure
from geo import neighborhood_procedure
from communities import louvain_communities_wrapper, determine_city_parts_from_redges
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
    G_nx: nx.Graph = ox.load_graphml(graphml_path[2])
    # weigths = nx.get_edge_attributes(G_nx, "weight")
    # new_weights = {}
    # for k, v in weigths.items():
    #     new_weights[k] = int(v)
    # G_nx = G_nx.to_undirected()
    G = Graph(json=kp_paths[9])#, nx=G_nx)

    # attack(G, 50, 'data/robust/bigattacks10-1000/test.json', 'cluster', False, True, ncuts=10, imb=0.05)

    with open('data/clusters/cluster1000_i01_t01.json', 'r') as rfile:
        clusters = json.load(rfile)
    # for cluster in clusters:
    #     print(len(cluster))
    # for i in range(0, len(clusters), 4):
    #     visualize_cluster(G_nx, clusters[i:i+4], 'data/anothercluster'+ str(i) +'.pdf')
    # s, s2 = 0, 0
    # for k, cls in enumerate(clusters):
    #     s += len(cls)
    #     for i, c1 in enumerate(cls):
    #         for j, c2 in enumerate(cls):
    #             if i > j and c1 == c2:
    #                 # print(f"dans le cls {k}, {i} = {j}")
    #                 s2 += 1
    #     print(s2)
    # print(s, s2)
    cls1 = clusters[3] + clusters[15] + clusters[23] + clusters[33]
    cls2 = clusters[0] + clusters[16] + clusters[22] + clusters[35] + clusters[36]
    cls3 = clusters[7] + clusters[27] + clusters[28] + clusters[31] + clusters[32] + clusters[39] + clusters[40] + clusters[41] + clusters[42]
    cls4 = clusters[1] + clusters[4] + clusters[5] + clusters[6] + clusters[8] + clusters[10] + clusters[24]
    cls5 = clusters[2] + clusters[21] + clusters[34]
    cls6 = clusters[9] + clusters[11] + clusters[26] + clusters[37] + clusters[38]
    cls7 = clusters[12] + clusters[13] + clusters[14] + clusters[17] + clusters[18] + clusters[19] + clusters[20] + clusters[29] + clusters[30]
    cls8 = clusters[25]
    clss = [cls1, cls2, cls3, cls4, cls5, cls6, cls7, cls8]
    # visualize_cluster(G_nx, [cls1, cls2, cls3, cls6 + cls7 + cls8], 'data/clusters.pdf')
    for i, cls in enumerate(clss):
        print(f"for cluster {i+1} of {len(cls)} cuts or {len(cls) * 1000 / 1518}")
        cluster_data_procedure(cls, G_nx)
        visualize_cluster(G_nx, [cls], 'data/cluster' + str(i) + '.pdf')

main()
