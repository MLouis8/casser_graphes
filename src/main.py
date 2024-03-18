from cuts_analysis import cpt_cuts_correlation, to_Cut, representant_method
from Graph import Graph
from visual import display_best_n_freq, visualize_class, nbclass_maxclass_plot
import osmnx as ox
import json
import numpy as np
import matplotlib.pyplot as plt

# f = {}
# for k, v in freq.items():
#     f[eval(k)] = v

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
    # cut_object_paths = [
    #     "./data/cuts_objects_01.json",
    #     ""]
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
    cuts = {}
    print("converting cuts...")
    for k, (_, blocks) in kcuts.items():
        cuts[k] = to_Cut(G_kp["xadj"], G_kp["adjncy"], blocks)
    print("classifying...")
    classes = representant_method(cuts, p=10, criterion="neighbor", G_kp=G_kp)
    with open("./data/rpz_nghbr3_003.json", "w") as write_file:
        json.dump(classes, write_file)

    # n_bins = 10
    
    # dist1 = [len(cls) for cls in classes]
    # _, ax = plt.subplots()
    c = []
    for i, cls in enumerate(classes):
        # if len(cls) > 10:
        #     c.append(i)
        print(i, len(cls))
    # print("displaying...")
    # print(len(c))
    # fig, axes = plt.subplots(3, 3)
    # for i in range(len(c)):
    #     visualize_class(classes[c[i]], G_nx, cuts, figsize=(3, 3), ax=axes[i//3, i%3], show=False)
    #     axes[i//3, i%3].set_title("classe de taille " + str(len(classes[i])))
    # fig.savefig("./presentations/images/visual_inter_rpz.pdf")

main()
