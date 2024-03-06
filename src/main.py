import osmnx as ox
import utils
import kahip
import matplotlib.pyplot as plt

filepaths = [
    "./data/Marseille_tol"+str(2**i)+".graphml" for i in range(5)
]
filepaths.append(
    "./data/Marseille_tol10.graphml"
)
#t = [2**i for i in range(5)]
#t.append(10)
#place = "Marseille, France"
#for i, filepath in enumerate(filepaths):
#    G = ox.graph_from_place(place, network_type="drive", clean_periphery=True)
#    G_proj = ox.project_graph(G)
#    G = ox.consolidate_intersections(G_proj, rebuild_graph=True, tolerance=t[i], dead_ends=False)
#    ox.save_graphml(G, filepath)
#G = ox.load_graphml(filepath)

def main():
     imbalances = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
     nb_blocks = [2**i for i in range(1, 6)]
     nb_trials = [2**i for i in range(1, 6)]
     
     for filepath in filepaths[0]:
         G_nx = ox.load_graphml(filepath)
         G_kp = utils.nx_to_kahip(G_nx)
         utils.preprocessing(G_kp)
         cut = kahip.kaffpa(G_kp, 2, 0.03, 0, rd.random(), 0)
         G_res = utils.rebuild(cut)
         utils.display_results(G_res, cut)
     
     filepath = ""
     results = []
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
     return
