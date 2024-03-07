import osmnx as ox

# Resultats observes lors de l'execution
# Just after importation, we have : 
# 94783 edges
# 70263 nodes
# After consolidation, we have : 
# 59060 edges
# 40547 nodes
# After projection, we have : 
# 59060 edges
# 40547 nodes

def init_city_graph(filepath):

    G = ox.graph_from_place('Paris, Paris, France', network_type="drive", buffer_dist=350,simplify=False,retain_all=True,clean_periphery=False,truncate_by_edge=False)
    G_Paris = ox.project_graph(G, to_crs='epsg:2154') ## pour le mettre dans le même référentiel que les données de Paris

    print('Just after importation, we have : ')
    print(str(len(G.edges())) + ' edges')
    print(str(len(G.nodes()))+ ' nodes')
    G2 = ox.consolidate_intersections(G_Paris, rebuild_graph=True, tolerance=4, dead_ends=True)
    print('After consolidation, we have : ')
    print(str(len(G2.edges())) + ' edges')
    print(str(len(G2.nodes()))+ ' nodes')
    G_out = ox.project_graph(G2, to_crs='epsg:4326')
    print('After projection, we have : ')
    print(str(len(G_out.edges())) + ' edges')
    print(str(len(G_out.nodes()))+ ' nodes')
    ox.save_graphml(G_out, filepath=filepath)

# init_city_graph("./data/Paris.graphml")