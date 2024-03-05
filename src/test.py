s = [i for i in range(10) if i%2 == 0]
print(s)

import osmnx as ox
G = ox.graph_from_place('Modena, Italy')
ox.plot_graph(G)