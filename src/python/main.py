from Graph import Graph
from cuts_analysis import determine_edge_frequency, class_mean_cost, get_n_biggest_freq
from utils import thousand_cuts, prepare_instance, preprocessing
from CutsClassification import CutsClassification
from visual import (
    visualize_class,
    basic_stats_cuts,
    basic_stats_edges,
    display_best_n_freq,
    visualize_edgeList,
    bar_plot,
)
from robustness import edge_frequency_attack, betweenness_attack
from paths import graphml_path, kp_paths, cut_paths_1, cut_paths_2, clusters_paths_2, clusters_paths_3

import osmnx as ox
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sys import setrecursionlimit

setrecursionlimit(100000)


# Code samples for main
def cpt_freq(freq, kcuts, G_kp):
    # necessaire pour l'import de fréquences
    f = {}
    for k, v in freq.items():
        f[eval(k)] = v
    # nécessaire pour la traduction des coupes
    cuts = {}
    for k, (edgecut, blocks) in kcuts.items():
        G_kp.set_last_results(edgecut, blocks)
        cuts[k] = G_kp.process_cut()


def clustering_procedure():
    print("import stuff...")
    G_nx = ox.load_graphml(graphml_path[2])
    G_kp = Graph(json=kp_paths[11])
    with open(cut_paths_2[2], "r") as read_file:
        kcuts = json.load(read_file)
    cuts = {}
    for k, (edgecut, blocks) in kcuts.items():
        G_kp.set_last_results(edgecut, blocks)
        cuts[k] = G_kp.process_cut()

    print("clustering...")
    C = CutsClassification(cuts, G_nx)
    n = 7500
    C.cluster_louvain("sum", n)
    print(f"for n = {n}")
    for level in C._levels:
        print(len(level))
    C.save_last_classes("data/clusters/CTS_" + str(n) + "_lanesmaxspeed.json")

def clustering_display():
    print("loading graphs...")
    j = 3
    G_nx = ox.load_graphml(graphml_path[2])
    G_kp = Graph(json=kp_paths[j+9])
    print("loading cuts and clusters...")
    with open(cut_paths_2[j], "r") as read_file:
        kcuts = json.load(read_file)
    cuts = {}
    for k, (edgecut, blocks) in kcuts.items():
        G_kp.set_last_results(edgecut, blocks)
        cuts[k] = G_kp.process_cut()
    with open(clusters_paths_3[7], "r") as read_file:
        levels = json.load(read_file)
    for level in levels:
        print(len(level))
    print("displaying...")
    fig, axes = plt.subplots(3, 4)
    fig.suptitle("clusters graphe valué par le nombre de voies sans pont")
    x, y = 1, 4
    for i in range(len(levels[x])):
        print(f"displaying axe {i}")
        visualize_class(levels[x][i], G_nx, cuts, ax=axes[i // y, i % y], show=False)
        axes[i // y, i % y].set_title(
            "taille: "
            + str(len(levels[x][i]))
            + ", coût moyen: "
            + str(round(class_mean_cost(levels[x][i], cuts, G_nx))),
            fontsize=6
        )
    axes[-1, -1].axis("off")
    axes[-1, -2].axis("off") 
    plt.savefig("presentations/images/clusters/CTS_lanesnobridge7500.pdf")


def main():
    G_bc = Graph(json=kp_paths[9]).to_nx()
    with open("data/robustness/ParisBCLanes.json", "w") as save:
        json.dump(nx.edge_betweenness_centrality(G_bc, weight="weight"), save)
main()
