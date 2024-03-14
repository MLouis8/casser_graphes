import osmnx as ox
from Graph import Graph, determine_edge_frequency
import matplotlib.pyplot as plt
from utils import cpt_cuts_correlation, preprocessing
import json
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

def freq_distributions(freq):
    total_edges = 46761
    n_bins = 25
    dist1 = freq.values()
    
    _, axs = plt.subplots(1, 2, tight_layout=True)

    for i in range(total_edges - len(dist1)):
        freq[str(i) + "vnnrs"] = 0
    dist2 = freq.values()
    axs[0].hist(dist1, bins=n_bins)
    axs[1].hist(dist2, bins=n_bins)
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    plt.savefig("./presentations/images/distribution_01.svg")

def main():
    # a excute a partir du repo Casser_graphe (chemin relatifs)
    kp_path = "./data/Paris.json"
    grahml_path = "./data/Paris.graphml"
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
    freqs = []
    with open(freq_paths[0], "r") as read_file:
        freqs.append(json.load(read_file))
    with open(freq_paths[1], "r") as read_file:
        freqs.append(json.load(read_file))
    with open(freq_paths[2], "r") as read_file:
        freqs.append(json.load(read_file))
    freq_distributions(freqs[0])
#PearsonRResult(statistic=0.10247828319260989, pvalue=1.094598985130352e-11)
main()
