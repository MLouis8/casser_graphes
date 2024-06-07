from Graph import Graph
from paths import graphml_path, kp_paths
from robustness import attack, cpt_effective_resistance
from visual import cumulative_impact_comparison
from procedures import compare_scc_or_cc_procedure, effective_resistance_procedure
from geo import neighborhood_procedure
from communities import louvain_communities_wrapper

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
    # G_nx = G_nx.to_undirected()
    G = Graph(json=kp_paths[9])#, nx=G_nx)
    # w = nx.get_edge_attributes(G.to_nx(), 'weight')
    # new_w = {}
    # for k, v in w.items():
    #     new_w[k] = eval(v)
    # nx.set_edge_attributes(G_nx, new_w, 'weight')
    
    # attack(G, 200, 'data/robust/bigattacks10-1000/deg200.json', 'deg', False, True, False)#, ncuts=10, imb=0.05, nblocks=4)

    # effective_resistance_procedure(G_nx, [], 'data/freq_gnx_er_base.json', True)

    paths = [
        'data/robust/bigattacks10-1000/bc_approx200.json',
        'data/robust/bigattacks10-1000/freq200_k2_01.json',
        'data/robust/bigattacks10-1000/freq200_k2_02.json',
        'data/robust/bigattacks10-1000/freq200_k2_03.json',
        'data/robust/bigattacks10-1000/freq200_k2_005.json',
        'data/robust/bigattacks10-1000/freq200_k3_005.json',
        'data/robust/bigattacks10-1000/freq200_k4_005.json',
        'data/robust/bigattacks10-1000/deg200.json',
        'data/robust/bigattacks10-1000/rd200.json'
    ]
    labels = [
        'bc', 'freq k=2 i=0.1', 'freq k=2 i=0.2', 'freq k=2 i=0.3', 'freq k=2 i=0.05', 'freq k=3 i=0.05', 'freq k=4 i=0.05', 'deg', 'rd'
    ]
    ccs = []
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for i in range(9):
        with open(paths[i], 'r') as rfile:
            data = json.load(rfile)
        ccs.append([e[1] for e in data])
    
    ax[0].plot(np.arange(len(ccs[4])), ccs[4], label=labels[4])
    ax[0].plot(np.arange(len(ccs[1])), ccs[1], label=labels[1])
    ax[0].plot(np.arange(len(ccs[2])), ccs[2], label=labels[2])
    ax[0].plot(np.arange(len(ccs[3])), ccs[3], label=labels[3])
    ax[0].legend()
    ax[0].set_title('frequency strategy with different imbalances')

    ax[1].plot(np.arange(len(ccs[4])), ccs[4], label=labels[4])
    ax[1].plot(np.arange(len(ccs[5])), ccs[5], label=labels[5])
    ax[1].plot(np.arange(len(ccs[6])), ccs[6], label=labels[6])
    ax[1].legend()
    ax[1].set_title('frequency strategy with different nblocks')

    ax[2].plot(np.arange(len(ccs[0])), ccs[0], label=labels[0])
    
    ax[2].plot(np.arange(len(ccs[5])), ccs[5], label=labels[5])
    ax[2].plot(np.arange(len(ccs[7])), ccs[7], label=labels[7])
    ax[2].plot(np.arange(len(ccs[3])), ccs[3], label=labels[3])
    ax[2].plot(np.arange(len(ccs[8])), ccs[8], label=labels[8])
    ax[2].legend()
    ax[2].set_title('comparison with classic strategies')
    fig.savefig('data/ccs.pdf')
        

main()
