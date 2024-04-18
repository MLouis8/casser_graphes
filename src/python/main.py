from Graph import Graph
from paths import graphml_path, kp_paths, rpaths_bc
from visual import visualize_bc

import json
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from sys import setrecursionlimit

setrecursionlimit(100000)

def main():
    with open(rpaths_bc[0], "r") as bc_lanes_file:
        bc_lanes = json.load(bc_lanes_file)
    with open(rpaths_bc[1], "r") as bc_nocost_file:
        bc_nocost = json.load(bc_nocost_file)
    G = ox.load_graphml(graphml_path[0])# nocost
    visualize_bc(bc_nocost, G, "presentations/images/visu_bc_nocost.pdf")
main()
