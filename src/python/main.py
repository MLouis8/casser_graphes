from Graph import Graph
from cuts_analysis import to_Cut, determine_edge_frequency
from utils import thousand_cuts, prepare_instance, preprocessing
from CutsClassification import CutsClassification
from visual import visualize_class, basic_stats_cuts, basic_stats_edges, display_best_n_freq
from paths import graphml_path, kp_paths, cut_paths, clusters_paths_2

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
    # necessaire pour l'import de fréquences
    f = {}
    for k, v in freq.items():
        f[eval(k)] = v
    # nécessaire pour la traduction des coupes
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
    
    print("import stuff...")
    G_nx = ox.load_graphml(graphml_path[1])
    G_kp = Graph(json=kp_paths[4])
    
    with open(cut_paths[14], "r") as read_file:
        kcuts = json.load(read_file)
    cuts = {}
    for k, (_, blocks) in kcuts.items():
        cuts[k] = to_Cut(G_kp["xadj"], G_kp["adjncy"], blocks)

    print("clustering...")
    C = CutsClassification(cuts, G_nx)
    n = 50000
    C.cluster_louvain("sum", n)
    print(f"for n = {n}")
    for level in C._levels:
        print(len(level))
    C.save_last_classes("data/clusters/CTS_"+str(n)+"widthmaxspeed.json")

    # print("displaying...") 
    # with open(clusters_paths_2[4], "r") as read_file:
    #     levels = json.load(read_file)
    # for level in levels:
    #     print(len(level))
    # fig, axes = plt.subplots(2, 3)
    # fig.suptitle("clusters graphe non valué (t=30000)")
    # for i in range(5):
    #     visualize_class(levels[0][i], G_nx, cuts, ax=axes[i//3, i%3], show=False)
    #     axes[i//3, i%3].set_title("classe de taille " + str(len(levels[0][i])))
    # plt.savefig("presentations/images/CTS_nocost_500.pdf")
    # plt.show()
main()