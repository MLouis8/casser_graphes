import osmnx as ox
from Graph import Graph, determine_edge_frequency
import matplotlib.pyplot as plt
from utils import cpt_cuts_correlation, preprocessing
import json
import geopandas

def main():
    # a excute a partir du repo Casser_graphe (chemin relatifs)
    kp_path = "./data/Paris.json"
    json_path = "./data/1000_cuts_Paris.json"
    grahml_path = "./data/Paris.graphml"
    filepath = "./presentations/images/special_edges.png"
    save_path = "./data/betweenness_Paris.json"
    print("import graphs...")
    G_nx = ox.load_graphml(grahml_path)
    preprocessing(G_nx)
    gdf_edges, gdf_vertices = ox.graph_to_gdfs(G_nx)
    gdf_edges.loc[gdf_edges.index.get_level_values('key') == 0]
    # print(f"preprocessing the graph...")
    # preprocessing(G_nx, val="width")
    # print(f"Conversion into KaHIP format...")
    # G_kp = Graph(nx=G_nx)
    # print("compute betweenness...")
    # betweenness = G_kp.compute_edge_betweenness()
    # with open(save_path, "w") as save_file:
    #     json.dump(betweenness, save_file)
    # print("import cuts...")
    # with open(json_path, "r") as read_file:
    #     cuts = json.load(read_file)
    # print("compute frequency...")
    # freq = determine_edge_frequency(G_kp, cuts)
    # print(cpt_cuts_correlation(freq, betweenness))

main()
