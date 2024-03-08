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
    json_path = "./data/1000_cuts_Paris.json"
    json_path2 = "./data/Paris.json"
    json_path3 = "./data/frequency_1000_cuts_Paris.json"
    grahml_path = "./data/Paris.graphml"
    print(f"importing graphs...")
    G_kp = Graph(json=json_path2)
    G_nx = ox.load_graphml(grahml_path)
    with open(json_path3, "r") as read_file:
        freq = json.load(read_file)
    f = { eval(k): v for k, v in freq.items()}
    display_freq(G_kp, G_nx, f)
main()