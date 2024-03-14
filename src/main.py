import osmnx as ox
from Graph import Graph, determine_edge_frequency
import matplotlib.pyplot as plt
from utils import cpt_cuts_correlation, preprocessing
import json
import networkx as nx
import numpy as np
import random as rd

def imbalances(G_nx, G_kp):
    imbalances = np.linspace(0, 0.1, 30)
    used_seed = []
    mean, minimum, maximum = [], [], []
    for epsilon in imbalances:
        res = []
        print(f"start cuts with imb={epsilon}")
        for i in range(25):
            seed = rd.randint(0,1044642763)
            while seed in used_seed:
                seed = rd.randint(0,1044642763)
            used_seed.append(seed)
            G_kp.kaffpa_cut(2, epsilon, 0, seed, 2)
            res.append(G_kp._edgecut)
        mean.append(int(np.mean(res)))
        minimum.append(min(res))
        maximum.append(max(res))
    return imbalances, mean, minimum, maximum

def main():
    # a excute a partir du repo Casser_graphe (chemin relatifs)
    kp_path = "./data/Paris.json"
    grahml_path = "./data/Paris.graphml"
    save_path = "./data/imbalances_analysis.json"
    print("import graphs...")
    G_nx = ox.load_graphml(grahml_path)
    G_kp = Graph(json=kp_path)
    print("start cutting...")
    imb, mean, minimum, maximum = imbalances(G_nx, G_kp)
    print("start saving...")
    imb_s = list(imb)
    print(imb_s)
    with open(save_path, "w") as write_file:
        json.dump({
            "imbalances": imb_s,
            "mean": mean,
            "min": minimum,
            "max": maximum
        }, write_file)
    ax = plt.subplot(111)
    ax.plot(imb, mean)
    ax.plot(imb, maximum)
    ax.plot(imb, minimum)
    ax.set_title("Graph of 25 runs statistics over different imbalances values")
    ax.legend()
    ax.show()
#PearsonRResult(statistic=0.10247828319260989, pvalue=1.094598985130352e-11)
main()
