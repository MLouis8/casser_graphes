from Graph import Graph
from cuts_analysis import to_Cut, determine_edge_frequency
from utils import thousand_cuts, prepare_instance
from CutsClassification import CutsClassification
from visual import visualize_class, basic_stats_cuts, basic_stats_edges, display_best_n_freq
from paths import graphml_path, kp_paths, cut_paths

import osmnx as ox
import json
import numpy as np
import random as rd
import networkx as nx
import matplotlib.pyplot as plt
from sys import setrecursionlimit
from scipy.stats import pearsonr

setrecursionlimit(100000)

# Code samples for main
def cpt_freq(freq, kcuts, G_kp):
    f = {}
    for k, v in freq.items():
        f[eval(k)] = v
    cuts = {}
    for k, (_, blocks) in kcuts.items():
        cuts[k] = to_Cut(G_kp["xadj"], G_kp["adjncy"], blocks)

def clustering(cuts, G_nx, filepath):
    print("clustering...")
    C = CutsClassification(cuts, G_nx)
    C.cluster_louvain()
    C.save_last_classes(filepath[2])

    print("displaying...") 
    with open(filepath[2], "r") as read_file:
        levels = json.load(read_file)
    for level in levels:
        print(level)
    fig, axes = plt.subplots(2, 3)
    fig.suptitle("clusters avec distance sum et treshold (10000)")
    for i in range(4):
        visualize_class(levels[0][i], G_nx, cuts, ax=axes[i//2, i%2], show=False)
        axes[i//2, i%2].set_title("classe de taille " + str(len(levels[0][i])))
    plt.show()
    # fig.savefig("./presentations/images/clusters/cluster_t_10000sub.pdf")

def main():
    
    # print("import stuff...")
    G_nx = ox.load_graphml(graphml_path)
    G_kp = Graph(json=kp_paths[0])
    
    with open(cut_paths[5], "r") as read_file:
        kcuts = json.load(read_file)
    # cuts = {}
    # for k, (_, blocks) in kcuts.items():
    #     cuts[k] = to_Cut(G_kp["xadj"], G_kp["adjncy"], blocks)
    
    freq = determine_edge_frequency(G_kp, kcuts)
    f = {}
    for k, v in freq.items():
        f[eval(k)] = v
    display_best_n_freq(G_nx, f)
main()