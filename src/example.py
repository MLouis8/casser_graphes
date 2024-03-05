import osmnx as ox
G = ox.graph_from_place('Modena, Italy')
ox.plot_graph(G)