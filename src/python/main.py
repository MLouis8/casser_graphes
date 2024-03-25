from cuts_analysis import distance, to_Cut, cluster_louvain
from Graph import Graph
from visual import visualize_class, nbclass_maxclass_plot
import osmnx as ox
import json
import numpy as np
import matplotlib.pyplot as plt
from sys import setrecursionlimit
import networkx as nx
from math import inf

setrecursionlimit(100000)
# f = {}
# for k, v in freq.items():
#     f[eval(k)] = v
# cuts = {}
# for k, (_, blocks) in kcuts.items():
#     cuts[k] = to_Cut(G_kp["xadj"], G_kp["adjncy"], blocks)

def main():
    # a excute a partir du repo Casser_graphe (chemin relatifs)
    kp_path = "./data/Paris.json"
    grahml_path = "./data/Paris.graphml"
    btw_path = "./data/betweenness_Paris.json"
    freq_paths = [
        "./data/frequency_1000_cuts_Paris_01.json",
        "./data/frequency_1000_cuts_Paris_003.json",
        "./data/frequency_1000_cuts_Paris.json"
    ]
    cut_paths = [
        "./data/1000_cuts_Paris_01.json",
        "./data/1000_cuts_Paris_003.json",
        "./data/1000_cuts_Paris.json",
    ]
    print("import stuff...")
    G_nx = ox.load_graphml(grahml_path)
    G_kp = Graph(json=kp_path)
    
    # freqs = []
    # with open(freq_paths[0], "r") as read_file:
    #     freqs.append(json.load(read_file))
    # with open(freq_paths[1], "r") as read_file:
    #     freqs.append(json.load(read_file))
    # with open(freq_paths[2], "r") as read_file:
    #     freqs.append(json.load(read_file))
    with open(cut_paths[1], "r") as read_file:
        kcuts = json.load(read_file)
    # with open("./data/betweenness_Paris.json", "r") as read_file:
    #     b = json.load(read_file)
    cuts = {}
    for k, (_, blocks) in list(kcuts.items())[:2]:
        cuts[k] = to_Cut(G_kp["xadj"], G_kp["adjncy"], blocks)
    limits = [(inf, inf), (inf, inf), (0, 0), (0, 0)]
    for u, v, _ in G_nx.edges:
        if u < limits[0][0]:
            limits[0] = (u, v)
        if v < limits[1][1]:
            limits[1] = (u, v)
        if u > limits[2][0]:
            limits[2] = (u, v)
        if v > limits[3][1]:
            limits[3] = (u, v)
    print(limits)
    edge_c = [
        'red' if (u, v) == (40540, 6068) else '#54545430' for u, v, _ in G_nx.edges
    ]
    edge_w=[
        20 if (u, v) in limits else 0.5 for u, v, _ in G_nx.edges
    ]
    ox.plot_graph(
        G_nx,
        bgcolor="white",
        node_size=0.5,
        edge_color=edge_c,
        edge_linewidth=edge_w,
        node_color="#54545420",
    )
    # print("clustering...")
    # classes = cluster_louvain(cuts, G_nx)
    # for cls in classes:
    #     print(len(cls))
    #     print(cls)
    # print("displaying...")
    # visualize_class(classes[0], G_nx, cuts)
    # fig, axes = plt.subplots(3, 3)
    # for i in range(9):
    #     visualize_class(classes[i], G_nx, cuts, figsize=(3, 3), ax=axes[i//3, i%3], show=False)
    #     axes[i//3, i%3].set_title("classe de taille " + str(len(classes[i])))
    # plt.show()
    # fig.savefig("./presentations/images/visual_rpz_geo005.pdf")
main()