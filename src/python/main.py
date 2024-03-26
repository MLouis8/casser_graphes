from cuts_analysis import to_Cut
from Graph import Graph
import osmnx as ox
import json
import numpy as np
import matplotlib.pyplot as plt
from sys import setrecursionlimit
from utils import flatten
from CutsClassification import CutsClassification
from visual import visualize_class

setrecursionlimit(100000)
# f = {}
# for k, v in freq.items():
#     f[eval(k)] = v
# cuts = {}
# for k, (_, blocks) in kcuts.items():
#     cuts[k] = to_Cut(G_kp["xadj"], G_kp["adjncy"], blocks)
def clustering(cuts, G_nx, filepath):
    print("clustering...")
    C = CutsClassification(cuts, G_nx)
    C.cluster_louvain()
    C.save_last_classes(filepath[2])
    print("displaying...")
    for level in C.get_levels():
        print(len(flatten(level)))
        # print(flatten(level))
        print(level)


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
    class_paths = [
        "./data/cluster_sum_003.json",
        "./data/cluster_sum_01.json",
        "./data/cluster_sumsq_003.json",
        "./data/cluster_inter_01.json",
        "./data/cluster_inter_003.json",
        "./data/cluster_max_003.json"
    ]
    print("import stuff...")
    G_nx = ox.load_graphml(grahml_path)
    G_kp = Graph(json=kp_path)
    
    with open(cut_paths[1], "r") as read_file:
        kcuts = json.load(read_file)
    cuts = {}
    for k, (_, blocks) in list(kcuts.items())[:100]:
        cuts[k] = to_Cut(G_kp["xadj"], G_kp["adjncy"], blocks)

    # with open(class_paths[0], "r") as read_file:
    #     levels = json.load(read_file)
    # print("analyse classes...")
    # for level in levels:
    #     print(len(level))
        
    print("clustering...")
    C = CutsClassification(cuts, G_nx)
    C.cluster_louvain("max")
    C.save_last_classes(class_paths[0])

    # print("displaying...")
    # fig, axes = plt.subplots(2, 2)
    # for i in range(4):
    #     visualize_class(levels[0][i], G_nx, cuts, ax=axes[i//2, i%2], show=False)
    #     axes[i//2, i%2].set_title("classe de taille " + str(len(levels[0][i])))
    # fig.savefig("./presentations/images/clusters003_sum.pdf")
main()