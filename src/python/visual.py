import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random as rd
import osmnx as ox
import pandas as pd
import networkx as nx

from cuts_analysis import get_n_biggest_freq, to_Cut
from typ import Cuts, Edge, EdgeDict, KCuts

def imbalances_cut(G_kp):
    imbalances = np.linspace(0, 0.1, 30)
    used_seed = []
    mean, minimum, maximum = [], [], []
    for epsilon in imbalances:
        res = []
        print(f"start cuts with imb={epsilon}")
        for i in range(25):
            seed = rd.randint(0, 1044642763)
            while seed in used_seed:
                seed = rd.randint(0, 1044642763)
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
    for filepath in [
        "./data/1000_cuts_Paris_01.json",
        "./data/1000_cuts_Paris_003.json",
        "./data/1000_cuts_Paris.json",
    ]:
        with open(filepath, "r") as read_file:
            cuts = json.load(read_file)
            conv = {"batch": [], "mean": [], "max": [], "min": []}
            res, cpt = [], 0
            for i, (edge_cut, _) in enumerate(cuts.values()):
                if i == 100:
                    break
                res.append(edge_cut)
                conv["max"].append(max(res))
                conv["min"].append(min(res))
                conv["mean"].append(np.mean(res))
            data.append(conv)

    imb = ["0.1", "0.03", "0.01"]
    batch = np.arange(100)
    for i in range(3):
        axes[i // 2, i % 2].plot(batch, data[i]["mean"], label="mean")
        axes[i // 2, i % 2].plot(batch, data[i]["max"], label="max")
        axes[i // 2, i % 2].plot(batch, data[i]["min"], label="min")
        axes[i // 2, i % 2].set_title(f"imbalance = {imb[i]}")
    fig.suptitle("Graph of 1000 runs statistics over 3 imbalances values")
    fig.legend()
    plt.savefig("./presentations/images/3convergences1000coupes.svg")


def exampleBastien(G_nx: nx.Graph):
    edges = ox.graph_to_gdfs(G_nx, nodes=False)
    edge_types = edges["length"].value_counts()
    color_list = ox.plot.get_colors(n=len(edge_types), cmap="viridis")
    color_mapper = pd.Series(color_list, index=edge_types.index).to_dict()

    # get the color for each edge based on its highway type
    ec = [color_mapper[d["length"]] for u, v, k, d in G_nx.edges(keys=True, data=True)]

    cmap = plt.cm.get_cmap("viridis")
    norm = plt.Normalize(vmin=edges["length"].min(), vmax=edges["length"].max())
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    fig, ax = ox.plot_graph(
        G_nx, edge_color=ec, bgcolor="w", node_size=0, figsize=(12, 12), show=False
    )
    cb = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        orientation="horizontal",
        shrink=0.5,
    )
    cb.set_label("length", fontsize=20)
    fig.savefig("demo.png")


def freq_distributions(freq, total_edges=46761):
    n_bins = 25
    dist1 = freq.values()

    _, axs = plt.subplots(1, 2, tight_layout=True)

    for i in range(total_edges - len(dist1)):
        freq[str(i) + "vnnrs"] = 0
    dist2 = freq.values()
    axs[0].hist(dist1, bins=n_bins)
    axs[1].hist(dist2, bins=n_bins)
    axs[0].set_yscale("log")
    axs[1].set_yscale("log")
    plt.savefig("./presentations/images/distribution_01.svg")


def basic_stats_edges(freq_dict: EdgeDict, g_size=46761, nb_cuts=1000):
    most_cut = max(freq_dict, key=freq_dict.get)
    less_cut = min(freq_dict, key=freq_dict.get)
    values = list(freq_dict.values())
    mean = np.mean(values)
    std = np.std(values)

    print("Here some basic stats on this set of cut(s):")
    print(f"The most cut edge ({most_cut}) has been cut {freq_dict[most_cut]} times")
    print(f"The less cut edge has been cut {freq_dict[less_cut]} times")
    print(f"We have a mean of {mean}")
    print(f"We have an std of {std}")
    print(
        f"With {nb_cuts} cuts we have cut {len(freq_dict.keys())} different edges over {g_size}"
    )


def basic_stats_cuts(cuts: dict[str, KCuts], nb_cuts=1000):
    nb_edges_cut = [edgecut for edgecut, _ in cuts.values()]
    best_cut = min(nb_edges_cut)
    worst_cut = max(nb_edges_cut)
    mean = np.mean(nb_edges_cut)
    std = np.std(nb_edges_cut)
    nb_best_cut = nb_edges_cut.count(best_cut)

    print("Here some basic stats on the set of cuts:")
    print(f"The best cut cuts {best_cut} edges")
    print(
        f"It appears {nb_best_cut} out of {nb_cuts} meaning a frequency of {nb_best_cut/nb_cuts}"
    )
    print(f"The worst cut cuts {worst_cut} edges")
    print(f"For {nb_cuts} cuts we have a mean of {mean} cut edges")
    print(f"And a std of {std}")


def display_freq(
    G_kp,
    G_nx: nx.Graph,
    f,
    savefig=False,
    filepath=None,
    show=True,
    ax=None,
    figsize=None,
):
    def colorize(u, v):
        if (u, v) in f:
            if f[(u, v)] > 400:
                return "tab:brown"
            elif f[(u, v)] > 300:
                return "tab:purple"
            elif f[(u, v)] > 200:
                return "tab:red"
            elif f[(u, v)] > 100:
                return "tab:orange"
            elif f[(u, v)] > 50:
                return "y"
            else:
                return "g"
        else:
            return "#54545460"

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
        edge_alpha=None,
    )


def display_best_n_freq(
    G_nx: nx.Graph,
    f,
    n=10,
    savefig=False,
    filepath=None,
    show=True,
    ax=None,
    figsize=None,
):
    notable_edges = get_n_biggest_freq(f, n)
    print(notable_edges)

    def colorize(u, v):
        if (u, v) in notable_edges.keys():
            if notable_edges[(u, v)] > 500:
                return "red"
            elif notable_edges[(u, v)] > 400:
                return "tab:orange"
            else:
                return "y"
        else:
            return "#54545460"

    def thicken(u, v):
        if (u, v) in notable_edges.keys():
            if notable_edges[(u, v)] > 500:
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


def visualize_class(
    cls: Cuts,
    G_nx: nx.Graph,
    cuts: dict[Edge, Cuts],
    savefig: bool = False,
    filepath=None,
    show: bool = True,
    ax=None,
):
    edges_to_color = set()
    for cut_id, edges in cuts.items():
        if cut_id in cls:
            edges_to_color |= set(edges) #.add(edges)
    def colorize(u, v):
        if (u, v) in edges_to_color:
            return "r"
        else:
            return "#54545430"

    def thicken(u, v):
        if (u, v) in edges_to_color:
            return 4 #1
        else:
            return 1
        
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
        node_color="#54545420",
        edge_alpha=None,
    )

def visualize_cost_heatmap(G_nx: nx.Graph, gradient: list[str], savefig: str=None):
    """
    Takes as parameter the city graph processed according to the desired cost
    (edges weight parameter should be set to the corresponding value)
    Saves of shows the plot
    """
    weight = nx.get_edge_attributes(G_nx, "weight")
    def colorize(u, v):
        if weight[(u, v)]  <= 4:
            return "#001AFF"
        elif weight[(u, v)] <= 8:
            return "#BC00A0"
        elif  weight[(u, v)] <= 20:
            return "#EA0066"
        else:
            return "#FF3300"
    _, ax = plt.subplot()
    edge_color = [colorize(u, v) for u, v, _ in G_nx.edges]
    print("edges colorized, starting display...")
    show = False if savefig else True
    save = not show
    return ox.plot_graph(
        G_nx,
        bgcolor="white",
        node_size=0.5,
        edge_color=edge_color,
        edge_linewidth=1,
        save=save,
        filepath=savefig,
        show=show,
        ax=ax,
        node_color="#54545420",
    )
    