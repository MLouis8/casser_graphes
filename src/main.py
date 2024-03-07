import osmnx as ox
import utils
import random as rd
from Graph import Graph

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

def prepare_instance(filename):
    filepath_graph = "./data/"+filename+".graphml"
    filepath_kahip = "./data/"+filename+".json"
    print(f"Loading instance {filepath_graph}")
    G_nx = ox.load_graphml(filepath_graph)
    print(f"preprocessing the graph...")
    utils.preprocessing(G_nx)
    print(f"Conversion into KaHIP format...")
    G_kp = Graph(nx=G_nx)
    G_kp.save_graph(filepath_kahip)
    
def main():
    imbalances = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    nb_blocks = [2**i for i in range(1, 6)]
    nb_trials = [2**i for i in range(1, 6)]
    json_path = "./data/Paris.json"
    grahml_path = "./data/Paris.graphml"

    print(f"importing graphs...")
    G_kp = Graph(json=json_path)
    G_nx = ox.load_graphml(grahml_path)
    seed = rd.randint(0,1044642763) 
    print(f"operating the cut...")
    G_kp.kaffpa_cut(2, 0.03, 0, seed, 2)
    print(f"preparing the results...")
    G_kp.display_city_cut(G_nx)
     
    #  filepath = ""
    #  results = []
     #G_nx = ox.load_graphml(filepath)
     #G_kp = utils.nx_to_kahip(G_nx)
     #utils.preprocessing(G_kp)
     #for epsilon in imbalances[0]:
        #cut = kahip.kaffpa(G_kp, 2, epsilon, 0, rd.random(), 2)
        #G_res = utils.rebuild(cut)
        #utils.display_results(G_res, cut)
        #results.append(cut)

     #_, ax = plt.subplots()
     #ax.plot(imbalances, len(cut[0]))

main()