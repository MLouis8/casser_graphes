import osmnx as ox
from Graph import Graph, determine_edge_frequency
import matplotlib.pyplot as plt
from utils import cpt_cuts_correlation
import json

def main():
    # a excute a partir du repo Casser_graphe (chemin relatifs)
    kp_path = "./data/Paris.json"
    json_path = "./data/1000_cuts_Paris.json"
    grahml_path = "./data/Paris.graphml"
    filepath = "./presentations/images/special_edges.png"
    G_nx = ox.load_graphml(grahml_path)
    G_kp = Graph(json=kp_path)
    betweenness = G_kp.compute_edge_betweenness()
    with open(json_path, "r") as read_file:
        cuts = json.load(read_file)
    freq = determine_edge_frequency(G_kp, cuts)
    print(cpt_cuts_correlation(freq, betweenness))
main()
