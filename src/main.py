import osmnx as ox
import random as rd
from Graph import Graph, determine_edge_frequency
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from utils import prepare_instance, basic_stats_edges, display_freq, basic_stats_cuts
import kahip
import json
import math
from colour import Color
    
def main():
    # a excute a partir du repo Casser_graphe (chemin relatifs)
    kp_path = "./data/Paris.json"
    json_path = "./data/1000_cuts_Paris_01.json"
    grahml_path = "./data/Paris.graphml"
    
    imbalances1 = np.linspace(0, 0.15, 100)
    res1 = [360, 407, 387, 420, 381, 406, 379, 384, 365, 384, 367, 332, 375, 364, 395, 375, 332, 403, 332, 365, 332, 365, 336, 365, 328, 332, 365, 328, 328, 328, 332, 332, 332, 368, 332, 328, 332, 328, 332, 332, 328, 365, 332, 328, 328, 328, 332, 332, 397, 328, 332, 328, 328, 332, 328, 332, 332, 332, 328, 328, 332, 332, 332, 328, 332, 332, 328, 332, 332, 365, 328, 332, 332, 328, 397, 328, 328, 328, 332, 328, 328, 332, 336, 343, 387, 411, 332, 343, 361, 369, 369, 329, 347, 375, 363, 348, 343, 360, 343, 373]
    imbalances2 = np.linspace(0.15, 0.4, 50)
    res2 = [393, 351, 329, 339, 325, 347, 336, 379, 381, 384, 334, 340, 332, 329, 373, 326, 329, 330, 329, 325, 341, 365, 360, 349, 333, 400, 382, 337, 345, 331, 316, 344, 296, 345, 292, 292, 329, 337, 337, 379, 415, 288, 381, 292, 338, 345, 329, 288, 329, 292]

    fig, ax = plt.subplot((2, 1))
    ax[0].plot(imbalances1, res1)
    ax[1].plot(imbalances2, res2)
    plt.savefig(fname="./presentations/images/desequilibre.svg")
main()
