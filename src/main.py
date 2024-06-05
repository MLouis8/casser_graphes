from Graph import Graph
from paths import graphml_path, kp_paths, dir_paths, effective_res_paths, redges_paths, eff_paths_dir, bigattacks_paths, bigattacks_paths2
from robustness import attack, cpt_effective_resistance
from visual import cumulative_impact_comparison, visualize_bc, visualize_Delta_bc, visualize_city_parts, visualize_edgeList, compare_avgebc_efficiency
from procedures import compare_scc_procedure
from geo import neighborhood_procedure
from communities import louvain_communities_wrapper, walktrap_communities_wrapper, determine_cut_edges

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
    # G_nx = G_nx.to_undirected()
    G = Graph(json=kp_paths[9])#, nx=G_nx)
    # w = nx.get_edge_attributes(G.to_nx(), 'weight')
    # new_w = {}
    # for k, v in w.items():
    #     new_w[k] = eval(v)
    # nx.set_edge_attributes(G_nx, new_w, 'weight')
    
    attack(G, 200, 'data/robust/bigattacks10-1000/rd200.json', 'rd', False, True, False)#, ncuts=10, imb=0.05, nblocks=4)

main()
