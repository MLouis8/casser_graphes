import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random as rd
import osmnx as ox
import pandas as pd
from cuts_analysis import get_n_biggest_freq, to_Cut

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
    plt.savefig("./presentations/images/distribution_01.svg")

def basic_stats_edges(dictionary, g_size=46761, nb_cuts=1000):
    most_cut = max(dictionary, key=dictionary.get)
    less_cut = min(dictionary, key=dictionary.get)
    values = list(dictionary.values())
    mean = np.mean(values)
    std = np.std(values)

    print("Here some basic stats on this set of cut(s):")
    print(f"The most cut edge ({most_cut}) has been cut {dictionary[most_cut]} times")
    print(f"The less cut edge has been cut {dictionary[less_cut]} times")
    print(f"We have a mean of {mean}")
    print(f"We have an std of {std}")
    print(f"With {nb_cuts} cuts we have cut {len(dictionary.keys())} different edges over {g_size}")

def basic_stats_cuts(cuts, nb_cuts=1000):
    nb_edges_cut = [edgecut for edgecut, _ in cuts.values()]
    best_cut = min(nb_edges_cut)
    worst_cut = max(nb_edges_cut)
    mean = np.mean(nb_edges_cut)
    std = np.std(nb_edges_cut)
    nb_best_cut = nb_edges_cut.count(best_cut)

    print("Here some basic stats on the set of cuts:")
    print(f"The best cut cuts {best_cut} edges")
    print(f"It appears {nb_best_cut} out of {nb_cuts} meaning a frequency of {nb_best_cut/nb_cuts}")
    print(f"The worst cut cuts {worst_cut} edges")
    print(f"For {nb_cuts} cuts we have a mean of {mean} cut edges")
    print(f"And a std of {std}")

def display_freq(G_kp, G_nx, f, savefig=False, filepath=None, show=True, ax=None, figsize=None):
    def colorize(u, v):
        if (u, v) in f:
            if f[(u,v)] > 400:
                return 'tab:brown'
            elif f[(u,v)] > 300:
                return 'tab:purple'
            elif f[(u,v)] > 200:
                return 'tab:red'
            elif f[(u,v)] > 100:
                return 'tab:orange'
            elif f[(u,v)] > 50:
                return 'y'
            else:
                return 'g'
        else:
            return '#54545460'
    edge_color = [colorize(u, v) for u, v, _ in G_nx.edges]
    print("edges colorized, starting display...")
    show = False if savefig else show
    return ox.plot_graph(
        G_nx,
        bgcolor="white",
        node_size=1,
        edge_color=edge_color,
        edge_linewidth=1,
        save=savefig,
        filepath=filepath,
        show=show,
        dpi=1024,
        ax=ax,
        figsize=figsize,
        node_color="#54545460",
        edge_alpha=None
    )

def display_best_n_freq(G_nx, f, n=10, savefig=False, filepath=None, show=True, ax=None, figsize=None):
    notable_edges = get_n_biggest_freq(f, n)
    print(notable_edges)
    def colorize(u, v):
        if (u, v) in notable_edges.keys():
            if notable_edges[(u,v)] > 500:
                return 'red'
            elif notable_edges[(u,v)] > 400:
                return 'tab:orange'
            else:
                return 'y'
        else:
            return '#54545460'
    def thicken(u, v):
        if (u, v) in notable_edges.keys():
            if notable_edges[(u,v)] > 500:
                return 10
            else:
                return 7
        else:
            return 0.5

    edge_color = [colorize(u, v) for u, v, _ in G_nx.edges]
    edge_width = [thicken(u, v) for u, v, _ in G_nx.edges]
    print("edges colorized, starting display...")
    show = False if savefig else show
    return ox.plot_graph(
        G_nx,
        bgcolor="white",
        node_size=1,
        edge_color=edge_color,
        edge_linewidth=edge_width,
        save=savefig,
        filepath=filepath,
        show=show,
        dpi=1024,
        ax=ax,
        figsize=figsize,
        node_color="#54545460",
    )

def visualize_class(cls, G_nx, cuts, savefig=False, filepath=None, show=True, ax=None, figsize=None):
    edges_to_color = set()
    for cut_id, edges in cuts.items():
        if cut_id in cls:
            edges_to_color |= set(edges)
    def colorize(u, v):
        if (u, v) in edges_to_color:
            return "r"
        else:
            return '#54545430'
    def thicken(u, v):
        if (u, v) in edges_to_color:
            return 4
        else:
            return 0.5
    edge_color = [colorize(u, v) for u, v, _ in G_nx.edges]
    edge_width = [thicken(u, v) for u, v, _ in G_nx.edges]
    print("edges colorized, starting display...")
    show = False if savefig else show
    return ox.plot_graph(
        G_nx,
        bgcolor="white",
        node_size=0.5,
        edge_color=edge_color,
        edge_linewidth=edge_width,
        save=savefig,
        filepath=filepath,
        show=show,
        ax=ax,
        figsize=figsize,
        node_color="#54545420",
        edge_alpha=None
    )

def nbclass_maxclass_plot(cuts, G_kp):
    epsilons = np.linspace(0.1, 1, 10)
    y1, y2 = [], []
    for eps in epsilons:
        print(f"classifying for eps={eps}")
        classes = representant_method(cuts, eps, 4, "mixed", G_kp)
        biggest_cls = 0
        for cls in classes:
            if len(cls) > biggest_cls:
                biggest_cls = len(cls)
        y1.append(len(classes))
        y2.append(biggest_cls)
    fig = plt.figure()
    axes = fig.subplots(1, 2)
    axes[0].plot(epsilons, y1)
    axes[0].set_ylabel("nb classes")
    axes[1].plot(epsilons, y2)
    axes[1].set_ylabel("biggest class size")
    axes[0].set_xlabel("epsilon")
    axes[1].set_xlabel("epsilon")
    fig.suptitle("Classification results: representant method + intersection criterion")
    plt.savefig("./presentations/images/mixed4_rpz_plots_003.pdf")

def mosaic_of_classes(kcuts, G_kp, G_nx):
    cuts = {}
    print("converting cuts...")
    for k, (_, blocks) in kcuts.items():
        cuts[k] = to_Cut(G_kp["xadj"], G_kp["adjncy"], blocks)
    print("classifying...")
    classes = representant_method(cuts)
    c = [0, 1, 2, 4, 5, 8, 10, 14, 24]
    print("displaying...")
    fig, axes = plt.subplots(3, 3)
    for i, k in enumerate(c):
        visualize_class(classes[k], G_nx, cuts, figsize=(3, 3), ax=axes[i//3, i%3], show=False)
        axes[i//3, i%3].set_title("classe de taille " + str(len(classes[k])))
    fig.savefig("./presentations/images/visual_rpz_inter05.pdf")




    # for k, v in G_nx.nodes(data=True):
        # print(k, v)
    # cpt_w, cpt_maxs, cpt_oneway, cpt_lanes, cpt_bridges, cpt_tunnel = 0, 0, 0, 0, 0, 0
    # cpt_rev, cpt_high, cpt_access, cpt_ref, cpt_junction, cpt_service = 0, 0, 0, 0, 0, 0
    # cpt_edges = 0
    # for e in G_nx.edges(data=True):
    #     cpt_edges += 1
    #     for attribute in e[2].keys():
    #         if attribute == 'oneway':
    #             cpt_oneway += 1
    #         if attribute == 'maxspeed':
    #             cpt_maxs += 1
    #         if attribute == 'reversed':
    #             cpt_rev += 1
    #         if attribute == 'highway':
    #             cpt_high += 1
    #         if attribute == 'lanes':
    #             cpt_lanes += 1
    #         if attribute == 'tunnel':
    #             cpt_tunnel += 1
    #         if attribute == 'ref':
    #             cpt_ref += 1
    #         if attribute == 'width':
    #             cpt_w += 1
    #         if attribute == 'access':
    #             cpt_access += 1
    #         if attribute == 'bridge':
    #             cpt_bridges += 1
    #         if attribute == 'junction':
    #             cpt_junction += 1 
    #         if attribute == 'service':
    #             cpt_service += 1
    # print(f"nb edges: {cpt_edges}")
    # print(f"nb oneway: {cpt_oneway} -> {cpt_oneway/cpt_edges}")
    # print(f"nb access: {cpt_access} -> {cpt_access/cpt_edges}")
    # print(f"nb bridges: {cpt_bridges} -> {cpt_bridges/cpt_edges}")
    # print(f"nb highway: {cpt_high} -> {cpt_high/cpt_edges}")
    # print(f"nb junction: {cpt_junction} -> {cpt_junction/cpt_edges}")
    # print(f"nb service: {cpt_service} -> {cpt_service/cpt_edges}")
    # print(f"nb width: {cpt_w} -> {cpt_w/cpt_edges}")
    # print(f"nb reversed: {cpt_rev} -> {cpt_rev/cpt_edges}")
    # print(f"nb lanes: {cpt_lanes} -> {cpt_lanes/cpt_edges}")
    # print(f"nb bridge: {cpt_bridges} -> {cpt_bridges/cpt_edges}")
    # print(f"nb ref: {cpt_ref} -> {cpt_ref/cpt_edges}")
    # print(f"nb tunnel: {cpt_tunnel} -> {cpt_tunnel/cpt_edges}")
    # print(f"nb max speed: {cpt_maxs} -> {cpt_maxs/cpt_edges}")