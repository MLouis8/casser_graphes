import osmnx as ox
import random as rd
from Graph import Graph, determine_edge_frequency
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from utils import prepare_instance, basic_stats, display_freq
import kahip
import json
import math
from colour import Color
    
def main():
    # a excute a partir du repo Casser_graphe (chemin relatifs)
    json_path = "./data/Paris.json"
    grahml_path = "./data/Paris.graphml"
    C = {}
    print("import graph...")
    G_kp = Graph(json=json_path)
    collision = []
    for i in range(1000):
        seed = rd.randint(0,1044642763)
        while seed in collision:
            print("collision")
            seed = rd.randint(0,1044642763)
        collision.append(seed)
        print(f"operating cut {i}")
        G_kp.kaffpa_cut(2, 0.1, 0, seed, 2)
        C[str(i)] = (G_kp.get_last_results)
    with open(r"./data/1000_cuts_Paris_01.json", "w") as write_file:
        json.dump(C, write_file)
main()
