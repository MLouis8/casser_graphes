import osmnx as ox
import random as rd
from Graph import Graph
import matplotlib.pyplot as plt
    
def main():
    imbalances = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    nb_blocks = [2**i for i in range(1, 6)]
    nb_trials = [2**i for i in range(1, 6)]
    json_path = "./data/Paris.json"
    grahml_path = "./data/Paris.graphml"

    print(f"importing graphs...")
    G_kp = Graph(json=json_path)
    G_nx = ox.load_graphml(grahml_path)
    results = []
    for id_cut, epsilon in enumerate(imbalances):
        seed = rd.randint(0,1044642763) 
        print(f"operating cut {id_cut}...")
        G_kp.kaffpa_cut(2, epsilon, 0, seed, 2)
        results.append(G_kp._edgecut)

    print(f"preparing the results...")
    # G_kp.display_city_cut(G_nx)
    # G_kp.display_last_cut_results()
    _, ax = plt.subplots()
    ax.plot(imbalances, results)
    plt.show()

main()