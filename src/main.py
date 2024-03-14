import osmnx as ox
from Graph import Graph, determine_edge_frequency
import matplotlib.pyplot as plt
from utils import cpt_cuts_correlation, preprocessing
import json
import networkx as nx
import numpy as np
import random as rd

from matplotlib import colors
from matplotlib.ticker import PercentFormatter

# Create a random number generator with a fixed seed for reproducibility
rng = np.random.default_rng(19680801)

def imbalances(G_nx, G_kp):
    imbalances = np.linspace(0, 0.1, 30)
    used_seed = []
    mean, minimum, maximum = [], [], []
    for epsilon in imbalances:
        res = []
        print(f"start cuts with imb={epsilon}")
        for i in range(25):
            seed = rd.randint(0,1044642763)
            while seed in used_seed:
                seed = rd.randint(0,1044642763)
            used_seed.append(seed)
            G_kp.kaffpa_cut(2, epsilon, 0, seed, 2)
            res.append(G_kp._edgecut)
        mean.append(int(np.mean(res)))
        minimum.append(min(res))
        maximum.append(max(res))
    return imbalances, mean, minimum, maximum

def freq_distributions(freq):
    N_points = 100000
    n_bins = 25

    # Generate two normal distributions
    dist1 = rng.standard_normal(N_points)
    dist2 = 0.4 * rng.standard_normal(N_points) + 5

    _, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    # We can set the number of bins with the *bins* keyword argument.
    axs[0].hist(dist1, bins=n_bins)
    axs[1].hist(dist2, bins=n_bins)

    # Now we format the y-axis to display percentage
    axs[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))

def main():
    # a excute a partir du repo Casser_graphe (chemin relatifs)
    kp_path = "./data/Paris.json"
    grahml_path = "./data/Paris.graphml"
    save_path = "./data/imbalances_analysis.json"
    print("import graphs...")
    G_nx = ox.load_graphml(grahml_path)
    G_kp = Graph(json=kp_path)
    print("start cutting...")
    imb, mean, minimum, maximum = imbalances(G_nx, G_kp)
    print("start saving...")
    imb_s = list(imb)
    print(imb_s)
    with open(save_path, "w") as write_file:
        json.dump({
            "imbalances": imb_s,
            "mean": mean,
            "min": minimum,
            "max": maximum
        }, write_file)
    ax = plt.subplot(111)
    ax.plot(imb, mean)
    ax.plot(imb, maximum)
    ax.plot(imb, minimum)
    ax.set_title("Graph of 25 runs statistics over different imbalances values")
    ax.legend()
    ax.show()
#PearsonRResult(statistic=0.10247828319260989, pvalue=1.094598985130352e-11)
main()
