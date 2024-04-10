from Graph import Graph
from cuts_analysis import to_Cut, determine_edge_frequency, class_mean_cost
from utils import thousand_cuts, prepare_instance, preprocessing
from CutsClassification import CutsClassification
from visual import visualize_class, basic_stats_cuts, basic_stats_edges, display_best_n_freq, visualize_edgeList
from paths import graphml_path, kp_paths, cut_paths_1, clusters_paths_2

import itertools
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
    cost = class_mean_cost(levels[0][i], cuts)
    for i in range(4):
        visualize_class(levels[0][i], G_nx, cuts, ax=axes[i//2, i%2], show=False)
        axes[i//2, i%2].set_title("(taille, coût moyen): (" + str(len(levels[0][i])) + ", " + str(cost) + ")")
    plt.show()
    # fig.savefig("./presentations/images/clusters/cluster_t_10000sub.pdf")

def main():
    imbalances = [0.03, 0.05, 0.1]
    paths = kp_paths[9:]
    c_names = ["lanes", "squared lanes", "lanes with maxspeed", "lanes without maxspeed"]
    for i, p in enumerate(paths):
        thousand_cuts(p, c_names[i], imbalances)
    thousand_cuts()
    
main()

# with open(cut_paths[8], "r") as read_file:
#     kcuts = json.load(read_file)
# cuts = {}
# for k, (_, blocks) in kcuts.items():
#     cuts[k] = to_Cut(G_kp["xadj"], G_kp["adjncy"], blocks)

# print("clustering...")
# C = CutsClassification(cuts, G_nx)
# n = 60000
# C.cluster_louvain("sum", n)
# print(f"for n = {n}")
# for level in C._levels:
#     print(len(level))
# C.save_last_classes("data/clusters/CTS_"+str(n)+"width.json")

# print("displaying...") 
# with open(clusters_paths_2[2], "r") as read_file:
#     levels = json.load(read_file)
# for level in levels:
#     print(len(level))
# fig, axes = plt.subplots(2, 2)
# fig.suptitle("clusters graphe valué par largeur sans tunnel")
# for i in range(4):
#     x, y = 0, 2 
#     visualize_class(levels[x][i], G_nx, cuts, ax=axes[i//y, i%y], show=False)
#     axes[i//y, i%y].set_title("classe de taille " + str(len(levels[x][i])))
# plt.savefig("presentations/images/clusters/CTS_widthnotunnel.pdf")
# plt.show()