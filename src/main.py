import osmnx as ox

filepath = "./data/Paris.graphml"
place = "Paris, France"
G = ox.graph_from_place(place, network_type="drive")
ox.save_graphml(G, filepath)
#G = ox.load_graphml(filepath)
