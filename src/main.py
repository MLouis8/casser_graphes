from cuts_analysis import cpt_cuts_correlation
from Graph import Graph
from visual import visualize_class, nbclass_maxclass_plot
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
    # with open(cut_paths[1], "r") as read_file:
    #     kcuts = json.load(read_file)
    

main()
