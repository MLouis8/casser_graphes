from cuts_analysis import to_Cut
from Graph import Graph
from visual import visualize_class, nbclass_maxclass_plot
import osmnx as ox
import json
import numpy as np
import matplotlib.pyplot as plt
from sys import setrecursionlimit
from utils import flatten
from CutsClassification import CutsClassification

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
    class_paths = [
        "./data/1000_cuts_Paris_003_cluster_sum.json",
        "./data/1000_cuts_Paris_01_cluster_sum.json",
        "./data/1000_cuts_Paris_003_cluster_sumsq.json"
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
    for k, (_, blocks) in list(kcuts.items()):
        cuts[k] = to_Cut(G_kp["xadj"], G_kp["adjncy"], blocks)

    print("clustering...")
    C = CutsClassification(cuts, G_nx)
    C.cluster_louvain()
    C.save_last_classes(class_paths[2])
    print("displaying...")
    for level in C.get_levels():
        print(len(flatten(level)))
        # print(flatten(level))
        print(level)
    # print(flatten(classes[1]))
    # print(flatten(classes[0]))
    # visualize_class(flatten(classes[1]), G_nx, cuts)
    # fig, axes = plt.subplots(3, 3)
    # for i in range(9):
    #     visualize_class(classes[i], G_nx, cuts, figsize=(3, 3), ax=axes[i//3, i%3], show=False)
    #     axes[i//3, i%3].set_title("classe de taille " + str(len(classes[i])))
    # plt.show()
    # fig.savefig("./presentations/images/visual_rpz_geo005.pdf")
main()