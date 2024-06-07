import networkx as nx
import osmnx as ox
import random as rd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from Graph import Graph
from typ import Edge, EdgeDict
from paths import (
    graphml_path,
    kp_paths,
    clusters_paths_3,
    cut_paths_2,
    dir_paths,
    redges_paths,
)
from visual import visualize_class, visualize_Delta_bc
from CutsClassification import CutsClassification
from cuts_analysis import class_mean_cost
from robustness import (
    extend_attack,
    efficiency,
    measure_scc_or_cc_from_rlist,
    cpt_effective_resistance,
)


def thousand_cuts_procedure(
    kp_paths: list[str], costs_names: list[str], imbalances: list[float], k: int = 2
) -> None:
    """
    Does 1000 cuts of each combination of parameters

    @params:
        Required: kp_paths, list of paths to KaHIP Graph save (json)
        Required: costs_names, list of costs names used for the produced files names
        Required: imbalances, list of imbalances values

        Optionnal: k, number of blocks to partition, by default set to 2

    @returns:
        None, saves the results in a new json file
    """
    assert len(kp_paths) == len(costs_names)
    for i, kp in enumerate(kp_paths):
        print(f"cutting for cost {costs_names[i]}...")
        for imbalance in imbalances:
            cut = {}
            seen_seeds = []
            print(f"cutting for imbalance {imbalance}...")
            for ncut in range(1000):
                print(
                    f"cut number {ncut} for imbalance {imbalance} and cost {costs_names[i]}"
                )
                G_kp = Graph(json=kp)
                seed = rd.randint(0, 1044642763)
                while seed in seen_seeds:
                    seed = rd.randint(0, 1044642763)
                seen_seeds.append(seed)
                G_kp.kaffpa_cut(k, imbalance, 0, seed, 2)
                cut[str(ncut)] = G_kp.get_last_results
            with open(
                "./data/cuts/" + costs_names[i] + "_1000_" + str(imbalance) + ".json",
                "w",
            ) as cut_file:
                json.dump(cut, cut_file)


# Code samples for main
def cpt_freq(freq, kcuts, G_kp):
    # necessaire pour l'import de fréquences
    f = {}
    for k, v in freq.items():
        f[eval(k)] = v
    # nécessaire pour la traduction des coupes
    cuts = {}
    for k, (edgecut, blocks) in kcuts.items():
        G_kp.set_last_results(edgecut, blocks)
        cuts[k] = G_kp.process_cut()


def clustering_procedure(
    graph_path: str, kp_path: str, cut_path: str, cost_name: str, treshold: int
):
    """ "
    @params:
        Required: graph_path, path to the OSMnx Graph
        Required: kp_path, path to the KaHIP Graph
        Required: cut_path, path to the cuts, it's a dictionnary of (cut id ex: "1", (edgecut, blocks))
        Required: cost_name, used only for saving file name
        Required: treshold, clustering treshold, if the distance between two cuts is smaller than the treshold, they are connected in the proximity graph.
    """
    print("import stuff...")
    G_nx = ox.load_graphml(graph_path)
    G_kp = Graph(json=kp_path)
    with open(cut_path, "r") as read_file:
        kcuts = json.load(read_file)
    cuts = {}
    for k, (edgecut, blocks) in kcuts.items():
        G_kp.set_last_results(edgecut, blocks)
        cuts[k] = G_kp.process_cut()

    print("clustering...")
    C = CutsClassification(cuts, G_nx)
    C.cluster_louvain(treshold)
    print(f"for n = {treshold}")
    for level in C._levels:
        print(len(level))
    C.save_last_classes(
        "data/clusters/CTS_" + str(treshold) + "_" + cost_name + ".json"
    )


def clustering_display_procedure():
    """Displays clusters"""
    print("loading graphs...")
    j = 3
    G_nx = ox.load_graphml(graphml_path[2])
    G_kp = Graph(json=kp_paths[j + 9])
    print("loading cuts and clusters...")
    with open(cut_paths_2[j], "r") as read_file:
        kcuts = json.load(read_file)
    cuts = {}
    for k, (edgecut, blocks) in kcuts.items():
        G_kp.set_last_results(edgecut, blocks)
        cuts[k] = G_kp.process_cut()
    with open(clusters_paths_3[7], "r") as read_file:
        levels = json.load(read_file)
    for level in levels:
        print(len(level))
    print("displaying...")
    fig, axes = plt.subplots(3, 4)
    fig.suptitle("clusters graphe valué par le nombre de voies sans pont")
    x, y = 1, 4
    for i in range(len(levels[x])):
        print(f"displaying axe {i}")
        visualize_class(levels[x][i], G_nx, cuts, ax=axes[i // y, i % y], show=False)
        axes[i // y, i % y].set_title(
            "taille: "
            + str(len(levels[x][i]))
            + ", coût moyen: "
            + str(round(class_mean_cost(levels[x][i], cuts, G_nx))),
            fontsize=6,
        )
    axes[-1, -1].axis("off")
    axes[-1, -2].axis("off")
    plt.savefig("presentations/images/clusters/CTS_lanesnobridge7500.pdf")


def extend_attack_procedure(prev_attack: str, saving_fp: str, **kwargs):
    """
    Procedure for simplifying the extension of an attack (continuing the removal of edges from what was already done)

    It requires the saving path of the original attack.
    And a new saving path for the resulting extension.
    All parameters of the first attack must be re-entered, only k the number of edges to remove is to change according to what's wanted.
    """
    with open(prev_attack, "r") as read_file:
        metrics = json.load(read_file)
    for i in range(len(metrics)):
        for j in range(len(metrics[i][0])):
            metrics[i][0][j] = eval(metrics[i][0][j])
    kwargs["ncuts"] = kwargs["ncuts"] if "ncuts" in kwargs else 1000
    kwargs["nrandoms"] = kwargs["nrandoms"] if "nrandoms" in kwargs else 100
    kwargs["save"] = kwargs["save"] if "save" in kwargs else True
    extend_attack(
        G=kwargs["G"],
        metrics=kwargs["metrics"],
        k=kwargs["k"],
        fp_save=kwargs["fp_save"],
        order=kwargs["order"],
        metric_bc=kwargs["metric_bc"],
        metric_cc=kwargs["metric_cc"],
        ncuts=kwargs["ncuts"],
        nrandoms=kwargs["nrandoms"],
        save=kwargs["save"],
    )


def bc_difference_map_procedure(
    i1: int,
    i2: int,
    read_fp: str,
    write_fp,
    graph_fp: str,
    order_name: str,
    abslt: bool,
):
    """Procedure for importing eBC values and displaying a city map colored following the absolute or relatives differences from one step to the other"""
    with open(read_fp, "r") as read_file:
        impt = json.load(read_file)
    G_nx = ox.load_graphml(graph_fp)
    bc1, bc2 = {}, {}
    for k, v in impt[i1][1].items():
        bc1[eval(k)] = v
    for k, v in impt[i2][1].items():
        bc2[eval(k)] = v
    try:
        r_edges = [eval(impt[j][0]) for j in range(i2 + 1)]
    except:
        r_edges = [
            (eval(impt[j][0][0]), eval(impt[j][0][1])) if impt[j][0] else impt[j][0]
            for j in range(i2 + 1)
        ]
    print(r_edges)
    visualize_Delta_bc(
        r_edges,
        bc1,
        bc2,
        G_nx,
        write_fp,
        abslt,
        "eBC diff map from "
        + str(i1)
        + "to "
        + str(i2)
        + " edges removed in "
        + order_name,
    )


def analyse_bcimpacts_procedure(
    robust_fps: list[str],
    eval_criterions: list[str],
    save_fp: str,
    names: list[str],
    titles: list[str],
):
    """Display dynamic impacts on a multiplot plot"""
    match len(eval_criterions):
        case 1:
            nlines, ncols = 1, 1
        case 2:
            nlines, ncols = 1, 2
        case 3:
            nlines, ncols = 1, 3
        case 4:
            nlines, ncols = 2, 2
        case _:
            raise ValueError("Only 1, 2, 3 or 4 evaluation criterions are allowed")
    fig, axes = plt.subplot(nlines, ncols)
    data = tuple([] for _ in range(len(eval_criterions)))
    for fp in robust_fps:
        with open(fp, "r") as read_file:
            robust_dicts = json.load(read_file)
        for i, criterion in enumerate(eval_criterions):
            data[i].append([dict[criterion] for dict in robust_dicts])
    x = np.arange(len(eval_criterions))
    if nlines == 1:
        for i in range(len(eval_criterions)):
            for j in range(len(fp)):
                axes[i].plot(x, data[i][j], label=names[j])
                axes[i].legend()
            axes[i].set_title(titles[i])
    else:
        for i in range(len(eval_criterions)):
            for j in range(len(fp)):
                axes[i // 2, i % 2].plot(x, data[i][j], label=names[j])
                axes[i // 2, i % 2].legend()
            axes[i // 2, i % 2].set_title(titles[i])
    fig.savefig(save_fp)


def efficiency_procedure(G_nx: nx.Graph, robust_path: str, efficiency_path: str):
    with open(robust_path, "r") as read_file:
        robustlist = json.load(read_file)
    efficiencies = []
    for attack in robustlist:
        efficiencies.append(efficiency(G_nx))
        if eval(attack[0]) and attack[0]:
            try:
                n1, n2 = eval(attack[0])[:2]
            except:
                n1, n2 = eval(attack[0][0]), eval(attack[0][1])
            try:
                G_nx.remove_edge(n1, n2)
            except:
                G_nx.remove_edge(n2, n1)
    with open(efficiency_path, "w") as save_file:
        json.dump(efficiencies, save_file)


def quality_bc_eval(
    real_bc: dict[str, float], bc_approxs: list[dict[str, float]]
) -> list[float]:
    corr = []
    for bc in bc_approxs:
        x, y = [], []
        for k, v in bc.items():
            x.append(v)
            y.append(real_bc[k])
        corr.append(pearsonr(x, y))
    return corr


def preprocess_robust_import(fp: str) -> tuple[list[Edge], list[EdgeDict]]:
    """Retrieves for a robust file save, removed edges and the different eBC dicts computed"""
    with open(fp, "r") as rfile:
        data = json.load(rfile)
    redges, bc_dicts = [None], []
    for attack in data:
        if attack[0] and attack[0] != "None":
            try:
                redges.append(eval(attack[0]))
            except:
                redges.append((eval(attack[0][0]), eval(attack[0][1])))
        d = {}
        for k, v in attack[1].items():
            d[eval(k)] = v
        bc_dicts.append(d)
    return (redges, bc_dicts)


def global_efficiency_procedure(
    G_nx: nx.Graph, robust_path: str, save_path: str, redges_import: bool
):
    """Computes the global efficiency for each Graph step, after each edge removal"""
    if redges_import:
        with open(robust_path, "r") as rfile:
            redges_file = json.load(rfile)
        redges = [eval(edge) for edge in redges_file]
    else:
        redges, _ = preprocess_robust_import(robust_path)
    with open(save_path, "w") as file:
        json.dump([], file)
    for edge in redges:
        if edge:
            G_nx.remove_edge(edge[0], edge[1])
        with open(save_path, "r") as file:
            globeff = json.load(file)
        globeff.append(nx.global_efficiency(G_nx))
        with open(save_path, "w") as file:
            json.dump(globeff, file)


def compare_scc_or_cc_procedure(
    G_nx: nx.Graph, robust_paths: list[str], labels: list[str], save_path: str, is_scc: bool
):
    """
    Measures every strongly connected components and plots the comparison of the biggest one size

    @params:
        Required: G_nx, the networkx graph
        Required: robust_paths, paths to the robust lists
        Required: labels, used only for the plots
        Required: save_plot, pdf or png path to save the produced plot
    """
    assert len(robust_paths) == len(labels)
    fig, ax = plt.subplots()
    for i, path in enumerate(robust_paths):
        print(f"scc of path {i}: {labels[i]}") if is_scc else print(f"cc of path {i}: {labels[i]}")
        G = G_nx.copy()
        with open(path, "r") as rfile:
            robust_list = json.load(rfile)
        y = measure_scc_or_cc_from_rlist(robust_list, G, is_scc)
        ax.plot(np.arange(len(y)), y, label=labels[i])
    ax.legend()
    ax.set_xlabel("number of removed edges")
    if is_scc:
        ax.set_ylabel("size of largest scc")
        fig.suptitle("largest scc evolution")
    else:
        ax.set_ylabel("size of largest cc")
        fig.suptitle("cc evolution")
    fig.savefig(save_path)


def effective_resistance_procedure(
    G_nx: nx.Graph,
    redges: list[tuple[int, int]],
    save_fp: str,
    weight: bool | None = None,
) -> None:
    """
    Launch the computation for effective resistance.
    It's a big computation, taking a lot of time.

    @params:
        Required: G_nx, the networkx graph on which the effective resistance will be computed
        Required: redges, the list of edges to remove before computing the effective resistance, leave empty if no edges have to be removed
        Required: save_fp, the path in which the computations are supposed to be stored.

        Optionnal: weight, boolean value on the use of the weights

    @returns:
        None
    """
    G = G_nx.to_undirected() if G_nx.is_directed() else G_nx
    if weight:
        w = nx.get_edge_attributes(G, "weight")
        new_w = {}
        for k, v in w.items():
            new_w[k] = eval(v)
        nx.set_edge_attributes(G, new_w, "weight")
    er_list = []
    with open(save_fp, "w") as wfile:
        json.dump(er_list, wfile)
    # test wether all removed edges are indeed in the Graph at the beginning
    for edge in redges:
        assert G.has_edge(edge[0], edge[1])
    # computes effective resistance
    for i, edge in enumerate(redges):
        G.remove_edge(edge[0], edge[1])
        try:
            er = (
                cpt_effective_resistance(G, True)
                if weight
                else cpt_effective_resistance(G, False)
            )
        except:
            print(
                f"an erreor occured while computing effective resistance, for edge {edge}, the {i}th over {len(redges)}"
            )
            raise ValueError("wrong")
        er_list.append(er)
        with open(save_fp, "w") as wfile:
            json.dump(er_list, wfile)


def verify_robust_list_integrity(path_index: int) -> None:
    """
    Does few checks to verify the integrity of the robust list created only for directed graphs.
    It's a code sample that can be easily adapted if needed.
    """

    def equal_dict(d1, d2) -> bool:
        if len(list(d1.keys())) != len(list(d2.keys())):
            return False
        for k, v in d1.items():
            if v != d2[k]:
                return False
        return True

    def assert_all_different_bcdicts(robust_list):
        for i, a1 in enumerate(robust_list):
            for j, a2 in enumerate(robust_list):
                if i == j:
                    continue
                if equal_dict(a1[1], a2[1]):
                    raise ValueError(
                        f"Dicts number {i} and {j} are the same ! (for the edges {a1[0]} and {a2[0]})"
                    )
        return True

    G = Graph(json=kp_paths[9])
    with open("data/cuts/lanes_1000_005.json", "r") as read_file:
        data = json.load(read_file)
    cut = data["141"]  # 190 or 24
    G.set_last_results(cut[0], cut[1])
    edges = G.process_cut()

    with open(dir_paths[path_index], "r") as rfile:
        data = json.load(rfile)
    with open(redges_paths[path_index], "r") as rfile:
        redges = json.load(rfile)
    for edge in redges:
        if not eval(edge) in edges:
            print("not in edge: ", edge)
    for edge in edges:
        if not str(edge) in redges:
            print("not in redges", edge)
    print(len(edges), len(redges))
    assert_all_different_bcdicts(data)
