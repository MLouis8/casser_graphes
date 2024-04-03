from Graph import Graph
from cuts_analysis import to_Cut
from utils import prepare_instance
from CutsClassification import CutsClassification
from visual import visualize_class
from progress_bar import printProgressBar

import osmnx as ox
import json
import numpy as np
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
    # a execute a partir du repo Casser_graphe (chemin relatifs)
    kp_paths = [
        # "./data/costs/nocost.json",
        "./data/costs/width.json",
        "./data/costs/widthsq.json",
        "./data/costs/widthmaxspeed.json",
        "./data/costs/widthnobridge.json",
        "./data/costs/widthnotunnel.json"
    ]
    graphml_path = "./data/Paris.graphml"
    btw_path = "./data/betweenness_Paris.json"
    freq_paths = [
        "./data/freqs/frequency_1000_cuts_Paris_01.json",
        "./data/freqs/frequency_1000_cuts_Paris_003.json",
        "./data/freqs/frequency_1000_cuts_Paris.json"
    ]
    cut_paths = [
        "./data/cuts/1000_cuts_Paris_01.json",
        "./data/cuts/1000_cuts_Paris_003.json",
        "./data/cuts/1000_cuts_Paris.json",
    ]
    class_paths = [
        "./data/clusters/cluster_sum_003.json",
        "./data/clusters/cluster_inter_01.json",
        "./data/clusters/cluster_inter_003.json",
        "./data/clusters/cluster_t_70000.json",
        "./data/clusters/cluster_t_50000.json",
        "./data/clusters/cluster_t_30000.json",
        "./data/clusters/cluster_t_10000.json"
    ]
    print("import stuff...")
    G_nx = ox.load_graphml(graphml_path)
    # G_kp = Graph(json=kp_paths[0])
    
    # with open(cut_paths[1], "r") as read_file:
    #     kcuts = json.load(read_file)
    # cuts = {}
    # for k, (_, blocks) in kcuts.items():
    #     cuts[k] = to_Cut(G_kp["xadj"], G_kp["adjncy"], blocks)

    costs = [
        # "no_val",
        "width",
        "squared width",
        "width with maxspeed",
        "width without bridge",
        "width without tunnel",
    ]
    for i, path in enumerate(kp_paths):
        print(f"path {path}")
        prepare_instance(graphml_path, path, costs[i])

main()