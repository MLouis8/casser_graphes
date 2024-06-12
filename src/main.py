from Graph import Graph
from paths import graphml_path, kp_paths, eff_paths_dir, bigattacks_paths, effective_res_paths
from robustness import attack, cpt_effective_resistance
from visual import cumulative_impact_comparison, visualize_edgeList_ordered, visualize_city_parts
from procedures import compare_scc_or_cc_procedure, effective_resistance_procedure
from geo import neighborhood_procedure
from communities import louvain_communities_wrapper, determine_city_parts_from_redges

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
    weigths = nx.get_edge_attributes(G_nx, "weight")
    new_weights = {}
    for k, v in weigths.items():
        new_weights[k] = int(v)
    # G_nx = G_nx.to_undirected()
    # G = Graph(json=kp_paths[9])#, nx=G_nx)
    # attack(G, 200, 'data/robust/bigattacks10-1000/deg200.json', 'deg', False, True, False)#, ncuts=10, imb=0.05, nblocks=4)

    # path = 'data/robust/bigattacks10-1000/bc_approx200.json'
    # with open(path, 'r') as rfile:
    #     data = json.load(rfile)
    # edges = [eval(a[0]) for a in data[1:174]]
    # parts = determine_city_parts_from_redges(G_nx.copy(), edges)
    # print([len(part) for part in parts])
    # visualize_city_parts(G_nx, parts, 'data/cutbcparts.pdf', edges)
    
main()
