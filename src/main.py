from Graph import Graph
from paths import graphml_path, kp_paths, eff_paths_dir, bigattacks_paths, effective_res_paths
from robustness import attack, cpt_effective_resistance
from visual import cumulative_impact_comparison, visualize_edgeList_ordered, visualize_city_parts, visualize_edgeList, visualize_cluster
from procedures import compare_scc_or_cc_procedure, effective_resistance_procedure, clustering_procedure
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

    # w = nx.get_edge_attributes(G_nx, 'weight')
    # new_w = {}
    # for k, v in w.items():
    #     new_w[k] = eval(v)
    # res = {}
    # for edge in G_nx.edges:
    #     res[edge] = new_w[edge] * G_nx.degree[edge[0]] * G_nx.degree[edge[1]]
    # G.set_weight_from_dict(res)

    # attack(G, 200, 'data/robust/bigattacks10-1000/dla200.json', 'dla', False, True)

    
    # with open('data/testCluster2600_03_mean.json', 'r') as rfile:
        # clusters = json.load(rfile)
    # for cluster in clusters:
    #     print(len(cluster))
    # visualize_cluster(G_nx, clusters, 'data/testCluster2600_03.pdf') # [clusters[2], clusters[3], clusters[5]]

    # clustering_procedure(G, G_nx, 'data/cuts/lanes_1000_03.json', 'data/testCluster2500_03.json', treshold=2500)

main()
