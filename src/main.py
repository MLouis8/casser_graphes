import osmnx as ox
import utils
import kahip
import matplotlib.pyplot as plt
import random as rd
import mes_modules
import networkx as nx

def main():
    imbalances = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    nb_blocks = [2**i for i in range(1, 6)]
    nb_trials = [2**i for i in range(1, 6)]
    filepath = "./data/Paris.graphml"
    seed = rd.randint(0,1044642763)
     
    print(f"Loading instance {filepath}")
    G_nx = ox.load_graphml(filepath)
    print(f"preprocessing the graph...")
    utils.preprocessing(G_nx)
    print(f"Conversion into KaHIP format...")
    vwght, xadj, adjcwgt, adjncy = utils.nx_to_kahip(G_nx)
    print(f"operating the cut...")
    cut = kahip.kaffpa(vwght, xadj, adjcwgt, adjncy, 2, 0.03, 0, seed, 2)
    print(f"preparing the results...")
    utils.display_results(G_nx, xadj, adjcwgt, adjncy, cut)
     
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