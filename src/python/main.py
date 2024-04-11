from Graph import Graph
from cuts_analysis import determine_edge_frequency, class_mean_cost
from utils import thousand_cuts, prepare_instance, preprocessing
from CutsClassification import CutsClassification
from visual import visualize_class, basic_stats_cuts, basic_stats_edges, display_best_n_freq, visualize_edgeList
from paths import graphml_path, kp_paths, cut_paths_1, cut_paths_2, clusters_paths_2

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

def clustering_procedure(cuts, G_nx, filepath):
    print("import stuff...")
    G_nx = ox.load_graphml(graphml_path[2])
    G_kp = Graph(json=kp_paths[10])
    with open(cut_paths_2[1], "r") as read_file:
        kcuts = json.load(read_file)
    cuts = {}
    for k, (edgecut, blocks) in kcuts.items():
        G_kp.set_last_results(edgecut, blocks)
        cuts[k] = G_kp.process_cut()

    print("clustering...")
    C = CutsClassification(cuts, G_nx)
    n = 70000
    C.cluster_louvain("sum", n)
    print(f"for n = {n}")
    for level in C._levels:
        print(len(level))
    C.save_last_classes("data/clusters/CTS_"+str(n)+"_lanessq.json")

    print("displaying...") 
    with open(clusters_paths_2[2], "r") as read_file:
        levels = json.load(read_file)
    for level in levels:
        print(len(level))
    fig, axes = plt.subplots(2, 2)
    fig.suptitle("clusters graphe valué par largeur sans tunnel")
    for i in range(4):
        x, y = 0, 2 
        visualize_class(levels[x][i], G_nx, cuts, ax=axes[i//y, i%y], show=False)
        axes[i//y, i%y].set_title("classe de taille " + str(len(levels[x][i])))
    plt.savefig("presentations/images/clusters/CTS_widthnotunnel.pdf")
    plt.show()

def main():
    G_kp = Graph(json=kp_paths[1])
    paths = cut_paths_2[4:]
    imbalances = [0., 0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.12, 0.16, 0.24, 0.32]
    distr_list = [{} for _ in imbalances]
    for i, p in enumerate(paths):
        with open(p, "r") as read_file:
            kcuts = json.load(read_file)
        for _, (edgecut, blocks) in kcuts.items():
            G_kp.set_last_results(edgecut, blocks)
            length = len([c for c in G_kp.get_connected_components(G_kp.to_nx())])
            if not length in distr_list[i].keys():
                distr_list[i][length] = 1
            else:
                distr_list[i][length] += 1

    with open("./data/freqs/cc_size_frequency_over_imb.json", "w") as write_file:
        json.dump(distr_list, write_file)

    for i, imb in enumerate(imbalances):
        print(f"for imbalance {imb} we have:")
        for k, v in distr_list[i]:
            print(f"    {k}: {v}")

main()