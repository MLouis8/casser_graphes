import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random as rd
import osmnx as ox
import pandas as pd

def imbalances(G_kp):
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

def triple_plot_convergence():
    fig = plt.figure()
    axes = fig.subplots(2, 2)
    data = []
    for filepath in ["./data/1000_cuts_Paris_01.json", "./data/1000_cuts_Paris_003.json", "./data/1000_cuts_Paris.json"]:
        with open(filepath, "r") as read_file:
            cuts = json.load(read_file)
            conv = {'batch': [], 'mean': [], 'max': [], 'min': []}
            res, cpt = [], 0
            for i, (edge_cut, _) in enumerate(cuts.values()):
                if i == 100:
                    break
                res.append(edge_cut)
                conv['max'].append(max(res))
                conv['min'].append(min(res))
                conv['mean'].append(np.mean(res))
            data.append(conv) 

    imb = ["0.1", "0.03", "0.01"]
    batch = np.arange(100)
    for i in range(3):
        axes[i//2, i%2].plot(batch, data[i]['mean'], label="mean")
        axes[i//2, i%2].plot(batch, data[i]['max'], label="max")
        axes[i//2, i%2].plot(batch, data[i]['min'], label="min")
        axes[i//2, i%2].set_title(f"imbalance = {imb[i]}")
    fig.suptitle("Graph of 1000 runs statistics over 3 imbalances values")
    fig.legend()
    plt.savefig("./presentations/images/3convergences1000coupes.svg")

def exampleBastien(G_nx):
    edges = ox.graph_to_gdfs(G_nx, nodes=False)
    edge_types = edges['length'].value_counts()
    color_list = ox.plot.get_colors(n=len(edge_types), cmap='viridis')
    color_mapper = pd.Series(color_list, index=edge_types.index).to_dict()

    # get the color for each edge based on its highway type
    ec = [color_mapper[d['length']] for u, v, k, d in G_nx.edges(keys=True, data=True)]

    cmap = plt.cm.get_cmap('viridis')
    norm=plt.Normalize(vmin=edges['length'].min(), vmax=edges['length'].max())
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    fig, ax = ox.plot_graph(G_nx, edge_color=ec,bgcolor='w',node_size=0, figsize=(12,12),show=False)
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='horizontal', shrink=0.5)
    cb.set_label('length', fontsize = 20)
    fig.savefig('demo.png')

def freq_distributions(freq):
    total_edges = 46761
    n_bins = 25
    dist1 = freq.values()
    
    _, axs = plt.subplots(1, 2, tight_layout=True)

    for i in range(total_edges - len(dist1)):
        freq[str(i) + "vnnrs"] = 0
    dist2 = freq.values()
    axs[0].hist(dist1, bins=n_bins)
    axs[1].hist(dist2, bins=n_bins)
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    plt.savefig("./presentations/images/distribution_003.svg")