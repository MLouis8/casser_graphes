import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random as rd
import osmnx as ox
import pandas as pd
import networkx as nx

from cuts_analysis import get_n_biggest_freq
from typ import Cuts, Edge, EdgeDict, KCut, RobustList


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


def exempleBastien(G_nx: nx.Graph):
    edges = ox.graph_to_gdfs(G_nx, nodes=False)
    edge_types = edges["length"].value_counts()
    color_list = ox.plot.get_colors(n=len(edge_types), cmap="viridis")
    color_mapper = pd.Series(color_list, index=edge_types.index).to_dict()

    # get the color for each edge based on its highway type
    ec = [color_mapper[d["length"]] for u, v, k, d in G_nx.edges(keys=True, data=True)]

    cmap = cm.get_cmap("viridis")
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


def basic_stats_cuts(cuts: dict[str, KCut], nb_cuts=1000):
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
            edges_to_color |= set(edges)  # .add(edges)

    def colorize(u, v):
        if (u, v) in edges_to_color:
            return "r"
        else:
            return "#54545430"

    def thicken(u, v):
        if (u, v) in edges_to_color:
            return 4  # 1
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


def visualize_edgeList(
    edgeList: list[Edge],
    G_nx: nx.Graph,
    thickness: EdgeDict | None = None,
    filepath: str | None = None,
    ax=None,
):
    def colorize(u, v, w):
        if (u, v) in edgeList or (u, v, w) in edgeList or (v, u) in edgeList or (v, u, w) in edgeList:
            return "r"
        else:
            return "#54545430"

    def thicken(u, v, w):
        if (u, v) or (v, u) in edgeList:
            return 10 # thickness[(u, v)] if (u, v) in edgeList else thickness[(v, u)]
        elif (u, v, w) or (v, u, w) in edgeList:
            return 10 # thickness[(u, v, w)] if (u, v, w) in edgeList else thickness[(v, u, w)]
        else:
            return 1

    edge_color = [colorize(u, v, w) for u, v, w in G_nx.edges]
    if bool(thickness):
        edge_width = [thicken(u, v, w) for u, v, w in G_nx.edges]
    else:
        edge_width = 1
    print("edges colorized, starting display...")
    return ox.plot_graph(
        G_nx,
        bgcolor="white",
        node_size=0.5,
        edge_color=edge_color,
        edge_linewidth=edge_width,
        save=bool(filepath),
        filepath=filepath,
        show=not bool(filepath),
        ax=ax,
        node_color="#54545420",
        edge_alpha=None,
    )


def visualize_cost_heatmap(
    G_nx: nx.Graph, gradient: list[str], savefig: str | None = None
):
    """
    Takes as parameter the city graph processed according to the desired cost
    (edges weight parameter should be set to the corresponding value)
    Saves of shows the plot
    """
    weight = nx.get_edge_attributes(G_nx, "weight")

    def colorize(u, v):
        if weight[(u, v)] <= 4:
            return "#001AFF"
        elif weight[(u, v)] <= 8:
            return "#BC00A0"
        elif weight[(u, v)] <= 20:
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


def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (_, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(
                x + x_offset,
                y,
                width=bar_width * single_width,
                color=colors[i % len(colors)],
            )

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    ax.legend(bars, data.keys())
    ax.set_ylabel("distribution")
    ax.set_xlabel("imbalances")


def to_hex(c):
    f = lambda x: int(255 * x)
    return "#{0:02x}{1:02x}{2:02x}".format(f(c[0]), f(c[1]), f(c[2]))


def visualize_bc(
    removed_edges: list[Edge], bc: EdgeDict, G: nx.Graph, fp: str, title: str, color_levels: int = 10, ax = None
) -> None:
    def colorize(u, v):        
        if (u, v) in bc:
            return color_list[int((bc[(u, v)] / vmax) * (color_levels-1))]
        elif (v, u) in bc:
            return color_list[int((bc[(v, u)] / vmax) * (color_levels-1))]
        # if (u, v, w) in bc:
        #     return color_list[int((bc[(u, v, w)] / vmax) * (color_levels-1))]
        # elif (v, u, w) in bc:
        #     return color_list[int((bc[(v, u, w)] / vmax) * (color_levels-1))]
        elif (u, v) in removed_edges or (v, u) in removed_edges:
            return "green"
        else:
            raise ValueError(f"edge ({u}, {v}) not in the graph")

    def thicken(u, v):
        if (u, v) in bc:
            return (2 * bc[(u, v)] / vmax) ** 2
        elif (v, u) in bc:
            return (2 * bc[(v, u)] / vmax) ** 2
        # if (u, v, w) in bc:
        #     return (2 * bc[(u, v, w)] / vmax) ** 2
        # elif (v, u, w) in bc:
        #     return (2 * bc[(v, u, w)] / vmax) ** 2
        elif (u, v) in removed_edges or (v, u) in removed_edges:
            return 5
        else:
            raise ValueError(f"edge ({u}, {v}) not in the graph")

    vmax, vmin = max(bc.values()), min(bc.values())
    # vmax, vmin = 0.1, 0
    color_list = [to_hex(elem) for elem in ox.plot.get_colors(color_levels, cmap="jet")] # RdPu for bicolor
    edge_color = [colorize(u, v) for u, v, _ in G.edges]
    edge_width = [thicken(u, v) for u, v, _ in G.edges]
    print("edges colorized, starting display...")
    cmap = plt.cm.get_cmap("jet") # RdPu for bicolor
    norm = plt.Normalize(vmin=0, vmax=0.1) #(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig, ax = ox.plot_graph(
        G,
        node_color="#54545420",
        node_size=0.5,
        edge_color=edge_color,
        edge_linewidth=edge_width,
        edge_alpha=0.7,
        bgcolor="white",
        show=False,
        ax=ax
    )
    cb = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation="horizontal", pad=0.01
    )
    cb.set_label(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(fp)


def visualize_Delta_bc(
    removed_edges: list[Edge],
    bc1: EdgeDict,
    bc2: EdgeDict,
    G: nx.Graph,
    fp: str,
    abslt: bool,
    title: str,
    color_levels: int = 10,
    ax = None
) -> None:
    def colorize(u, v):
        d = None
        if (u, v) in delta.keys():
            d = g(delta[(u, v)])
        elif (v, u) in delta.keys():
            d = g(delta[(v, u)])
        elif (u, v) in removed_edges or (v, u) in removed_edges:
            return "green"
        else:
            raise ValueError(f"edge ({u}, {v}) not in the graph")
        return color_list[int((d / (2*vmax)) * (color_levels-1))]

    def thicken(u, v):
        if (u, v) in delta.keys():
            return (g(delta[(u, v)]) / vmax) ** 2
        elif (v, u) in delta.keys():
            return (g(delta[(v, u)]) / vmax) ** 2
        elif (u, v) in removed_edges or (v, u) in removed_edges:
            return 5
        else:
            raise ValueError(f"edge ({u}, {v}) not in the graph")

    f = lambda b1, b2: abs(b1 - b2) if abslt else b2 - b1
    g = lambda v: v if abslt else v+vmax
    delta = {}
    for n1, n2, _ in G.edges:
        if (n1, n2) in bc1:
            if (n1, n2) in bc2:
                delta[(n1, n2)] = f(bc1[(n1, n2)], bc2[(n1, n2)])
            elif (n2, n1) in bc2:
                delta[(n1, n2)] = f(bc1[(n1, n2)], bc2[(n2, n1)])
        elif (n2, n1) in bc1:
            if (n1, n2) in bc2:
                delta[(n2, n1)] = f(bc1[(n2, n1)], bc2[(n1, n2)])
            elif (n2, n1) in bc2:
                delta[(n2, n1)] = f(bc1[(n2, n1)], bc2[(n2, n1)])
    if abslt:
        vmax, vmin = max(delta.values()), min(delta.values())
    else:
        m = max(abs(min(delta.values())), abs(max(delta.values())))
        if m == 0:
            vmin, vmax = -1e-7, 1e-7
        else:
            vmin, vmax = -0.1, 0.1 #-m, m
    if abslt:
        color_list = [to_hex(elem) for elem in ox.plot.get_colors(color_levels, cmap="RdPu")]
    else:
        color_list = [to_hex(elem) for elem in ox.plot.get_colors(color_levels, cmap="bwr")]
    edge_color = [colorize(u, v) for u, v, _ in G.edges]
    edge_width = [thicken(u, v) for u, v, _ in G.edges]
    print("edges colorized, starting display...")
    cmap = plt.cm.get_cmap("RdPu") if abslt else plt.cm.get_cmap("bwr")
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig, ax = ox.plot_graph(
        G,
        node_color="#54545420",
        node_size=0.5,
        edge_color=edge_color,
        edge_linewidth=edge_width,
        show=False,
        bgcolor="white",
        ax=ax
    )
    cb = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation="horizontal", pad=0
    )
    cb.set_label(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(fp)


def visualize_bc_distrs(bc1: EdgeDict, bc2: EdgeDict, fp: str, names: tuple[str, str], cumulative: bool = False) -> None:
    dist1 = bc1.values()
    dist2 = bc2.values()
    _, ax = plt.subplots()
    htype = "step" if cumulative else "bar"
    ax.hist(dist1, bins='scott', label=names[0], color="#ff0000a0", histtype=htype, cumulative=cumulative)
    ax.hist(dist2, bins='scott', label=names[1], color="#0000ffa0", histtype=htype, cumulative=cumulative)
    ax.set_yscale("log")
    ax.set_xlabel("bc")
    ax.set_ylabel("number of edges")
    ax.legend()
    plt.savefig(fp)


def visualize_Deltabc_distrs(
    bc1: EdgeDict, bc2: EdgeDict, fp: str, abslt: bool
) -> None:
    """
    Plots distribution of difference between two edge Betweenness Centrality dictionnaries
    asblt indicates whether the difference is to observe relatively or absolutely
    """
    n_bins = 25
    f = lambda b1, b2: abs(b1 - b2) if abslt else b2 - b1
    dist = [f(bc1[k], bc2[k]) for k in bc1.keys()]
    _, ax = plt.subplots()
    ax.hist(dist, bins=n_bins)
    ax.set_yscale("log")
    ax.legend()
    plt.savefig(fp)


def visualize_attack_scores(
    attacks: (
        list[list[list[float]] | list[EdgeDict]] | list[list[list[int]] | list[int]]
    ),
    attacks_names: list[str],
    fp: str,
    is_bc: bool,
    title: str
) -> None:
    """
    Function to visualize the evolution of metrics over multiple attacks
    is_bc indicates whether the observed scores are average eBC or biggest component size (if set to False)
    """
    fig, ax = plt.subplots()
    for i, attack in enumerate(attacks):
        x = np.arange(len(attack))
        if is_bc:
            if type(attack[0]) == dict:
                # attack is a list of eBCs
                y = [np.mean(list(bc_dict.values())) for bc_dict in attack]
                ax.plot(x, y, label="bc " + attacks_names[i])
            elif type(attack[0]) == list:
                # attack is a list of lists of avg eBCs (coming from random attack)
                y_moy = [np.mean(list(bc_dict.values())) for bc_dict in attack]
                y_min = [np.min(list(bc_dict.values())) for bc_dict in attack]
                y_max = [np.max(list(bc_dict.values())) for bc_dict in attack]
                ax.plot(x, y_moy, label="bc moy " + attacks_names[i])
                ax.plot(x, y_min, label="bc min " + attacks_names[i])
                ax.plot(x, y_max, label="bc max" + attacks_names[i])
            else:
                raise TypeError(f"Type {type(attack[0])} not recognized")
        else:
            if type(attack[0]) == list:
                # attack is a list of biggest size cc (coming from random attack)
                y_moy = [np.mean(cc) for cc in attack]
                y_min = [np.min(cc) for cc in attack]
                y_max = [np.max(cc) for cc in attack]
                ax.plot(x, y_moy, label="cc moy " + attacks_names[i])
                ax.plot(x, y_min, label="cc min " + attacks_names[i])
                ax.plot(x, y_max, label="cc max" + attacks_names[i])
            elif type(attack[0]) == int:
                # attack is a list of biggest size cc
                ax.plot(x, attack, label="scc " + attacks_names[i])
            else:
                raise TypeError(f"Type {type(attack[0])} not recognized")
    if is_bc:
        ax.set_ylabel("average edge BC")
    else:
        # plt.ylim(38450, 38650)
        # plt.autoscale(False)
        ax.set_ylabel("biggest strongly connected component size")
    ax.set_xlabel("number of removed edges")
    ax.legend()
    fig.suptitle(title)
    fig.savefig(fp)


def visualize_edgeList_ordered(edgeList: list[Edge], G: nx.Graph, fp: str, title: str) -> None:
    def colorize(u, v):        
        if (u, v) in edgeList:
            i = edgeList.index((u, v))
            return color_list[i]
        if (v, u) in edgeList:
            i = edgeList.index((v, u))
            return color_list[i]
        else:
            return "#54545420"

    def thicken(u, v):
        if (u, v) in edgeList or (v, u) in edgeList:
            return 5
        else:
            return 1
    color_list = [to_hex(elem) for elem in ox.plot.get_colors(len(edgeList), cmap="plasma")]
    edge_color = [colorize(u, v) for u, v, _ in G.edges]
    edge_width = [thicken(u, v) for u, v, _ in G.edges]
    print("edges colorized, starting display...")
    cmap = plt.cm.get_cmap("plasma")
    norm = plt.Normalize(vmin=0, vmax=len(edgeList))
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig, ax = ox.plot_graph(
        G,
        node_color="#54545420",
        node_size=0.5,
        edge_color=edge_color,
        edge_linewidth=edge_width,
        bgcolor="white",
        show=False,
    )
    cb = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation="horizontal", pad=0.01
    )
    cb.set_label(title, fontsize=12)
    fig.savefig(fp)

def visualize_biggest_scc(G_nx: nx.Graph, fp: str, robust_list: RobustList | None) -> None:
    """
    Saves in fp the city graph with biggest connected component highlighted
    If robust_list is set to None, it doesn't remove any edge from G_nx
    If a robust_list is given, it removes all corresponding edges (assumes it's the last attack)
    """
    def colorize(u, v):        
        if (u, v) in biggest_scc or (v, u) in biggest_scc:
            return "r"
        else:
            return "#54545420"
    if robust_list:
        for attack in robust_list:
            if eval(attack[0]) is None:
                continue
            try:
                u, v = eval(attack[0])
            except:
                if eval(attack[0][0]) is None:
                    continue
                u, v = eval(attack[0][0]), eval(attack[0][1])
            G_nx.remove_edge(u, v)
    biggest_scc = max(nx.strongly_connected_components(G_nx), key=len)
    edge_color = [colorize(u, v) for u, v, _ in G_nx.edges]
    fig, _ = ox.plot_graph(
        G_nx,
        node_color="#54545420",
        node_size=0.5,
        edge_color=edge_color,
        edge_linewidth=1,
        bgcolor="white",
        show=False,
    )
    fig.savefig(fp)

def visualize_impact_evolution(impact_path: str, impact_crit: str, title: str, save_path: str) -> None:
    with open(impact_path, "r") as rfile:
        data = json.load(rfile)
    x, y = np.arange(len(data)), []
    for impact in data:
        y.append(impact[impact_crit])
    fig, ax = plt.subplots()
    ax.plot(x, y, label=impact_crit)
    ax.set_ylabel("number of impacted edges")
    ax.set_xlabel("edge removal")
    ax.legend()
    fig.suptitle(title)
    fig.savefig(save_path)

def visualize_impacts_comparison(impact_paths: str | list[str], impact_crit: str | list[str], labels: str | list[str], title: str, save_path: str, cumulative: bool = False) -> None:
    fig, axes = plt.subplots()
    if isinstance(impact_paths, list):
        assert isinstance(impact_crit, str)
        for i, impact_path in enumerate(impact_paths):
            with open(impact_path, "r") as rfile:
                data = json.load(rfile)
            x, y = np.arange(len(data)), []
            for impact in data:
                y.append(impact[impact_crit])
            axes.plot(x, y, label=str(impact_crit) + " " + labels[i])
    else:
        with open(impact_paths, "r") as rfile:
            data = json.load(rfile)
        assert isinstance(impact_crit, list)
        for i, crit in enumerate(impact_crit):
            x, y = np.arange(len(data)), []
            for impact in data:
                y.append(impact[crit])
            axes.plot(x, y, label=str(crit) + " " + labels[i])
    axes.set_ylabel("number of impacted edges")
    axes.set_yscale("log")
    axes.set_xlabel("edge removal")
    axes.legend()
    fig.suptitle(title)
    fig.savefig(save_path)

def cumulative_impact_comparison(impacts_fp: list[str], crit: str, label: str, attack_names: list[str], save_fp: str):
    impacts = []
    for fp in impacts_fp:
        with open(fp, "r") as rfile:
            impact = json.load(rfile)
        impacts.append([d[crit] for d in impact])
    fig, ax = plt.subplots()
    for i, impact in enumerate(impacts):
        t = np.arange(len(impact))
        ax.plot(t, impact, label=attack_names[i])
    ax.legend()
    ax.set_ylabel(label)
    ax.set_xlabel("number of removed edges")
    fig.suptitle("Cumulative evolution of "+crit)
    fig.savefig(save_fp)

def compare_avgebc_efficiency(robust_paths: list[str], efficiency_paths: list[str], labels: list[str], save_path: str):
    colors = cm.get_cmap("hsv", len(labels))
    fig, ax1 = plt.subplots()
    for i, path in enumerate(robust_paths):
        with open(path, "r") as rfile:
            robust_dict = json.load(rfile)
        if labels[i] == "freq":
            x, y = np.arange(len(robust_dict))-1, [np.mean(list(attack[1].values())) for attack in robust_dict]
            ax1.plot(x, y[:len(robust_dict)-1])
        else:
            x, y = np.arange(len(robust_dict)), [np.mean(list(attack[1].values())) for attack in robust_dict]
            ax1.plot(x, y, label=labels[i], color=colors[i])
    ax2 = ax1.twinx()
    for i, path in enumerate(efficiency_paths):
        with open(path, "r") as rfile:
            efficiency = json.load(rfile)
        ax2.plot(np.arange(len(efficiency)), efficiency, label=labels[i], color=colors[i], linestyle='dotted')

    ax1.set_xlabel('number of removed edges')
    ax1.set_ylabel('(-) avg eBC')
    ax2.set_ylabel('efficiency (.)')
    ax1.legend()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig(save_path)