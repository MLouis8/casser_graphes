import osmnx as ox
import random as rd
from Graph import Graph
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from utils import prepare_instance
import kahip
import json
    
def main():
    # a excute a partir du repo Casser_graphe (chemin relatifs)
    json_path = "./data/Paris.json"
    grahml_path = "./data/Paris.graphml"
    print(f"importing graphs...")
    G_kp = Graph(json=json_path)
    G_nx = ox.load_graphml(grahml_path)
    C = {}
    collision = []
    for i in range(10000):
        seed = rd.randint(0,1044642763)
        while seed in collision:
            print("collision")
            seed = rd.randint(0,1044642763)
        print(f"operating cut {i}")
        G_kp.kaffpa_cut(2, 0.01, 0, seed, 2)
        C[str(i)] = (G_kp.get_last_results)
        #print(f"preparing the results...")
        #_, _ = G_kp.display_city_cut(G_nx, show=False, ax=axs[i//3, i%3], figsize=(3, 3))
    with open(r"./data/10000_cuts_Paris.json", "w") as write_file:
        json.dump(C, write_file)

   # plt.savefig(fname="./data/images/imbalances_paris_0,01-0,1.svg")
    #plt.savefig(fname="./data/images/imbalances_paris_0,01-0,1.png", dpi=1024)
    # G_kp.display_last_cut_results()
    # _, ax = plt.subplots()
    # ax.plot(imbalances, results)
    # plt.show()

main()