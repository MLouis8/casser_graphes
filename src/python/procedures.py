import networkx as nx
import osmnx as ox
import random as rd
import json
import math
import matplotlib.pyplot as plt

from Graph import Graph
from typ import EdgeDict3, Edge
from paths import graphml_path, kp_paths, clusters_paths_3, cut_paths_2
from visual import visualize_class
from CutsClassification import CutsClassification
from cuts_analysis import class_mean_cost
from robustness import extend_attack


def replace_parallel_edges(G):
    """
    KaHIP ne suppporte pas les aretes paralleles, on les remplace donc
    par un noeud qui sert d'intermediaire pour une nouvelle arete.
    """
    parallel_edges = [(u, v, k) for u, v, k in G.edges if k != 0]

    edges_weight = nx.get_edge_attributes(G, "weight")

    for edge in parallel_edges:
        u, v, k = edge[0], edge[1], edge[2]
        x_mid = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2
        y_mid = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2

        new_node = max(G.nodes) + 1
        G.add_node(new_node, x=x_mid, y=y_mid)
        G.add_edge(u, new_node, 0)
        G.add_edge(new_node, v, 0)

        # Si les aretes sont valuees alors les nouvelles en heritent
        if edges_weight:
            edges_weight[(u, new_node, 0)] = int(edges_weight[(u, v, k)])
            edges_weight[(new_node, v, 0)] = int(edges_weight[(u, v, k)])

        # Pareil pour les attributs
        G.edges[(u, v, 0)].update(G.edges[(u, v, k)])
        G.edges[(u, new_node, 0)].update(G.edges[(u, v, 0)])
        G.edges[(new_node, v, 0)].update(G.edges[(u, v, 0)])

        # Sauf pour la longueur qui est divisee par deux
        G.edges[(u, new_node, 0)]["length"] = G.edges[(u, v, 0)]["length"] / 2
        G.edges[(new_node, v, 0)]["length"] = G.edges[(u, v, 0)]["length"] / 2

        G.remove_edge(u, v, k)

    node_weights = {}
    for node in G.nodes:
        node_weights[node] = 1

    nx.set_node_attributes(G, node_weights, "weight")
    nx.set_edge_attributes(G, edges_weight, "weight")


def add_node_weights_and_relabel(G):
    w_nodes = {}
    for node in list(G.nodes):
        w_nodes[node] = 1
    nx.set_node_attributes(G, w_nodes, "weight")
    sorted_nodes = sorted(G.nodes())
    mapping = {old_node: new_node for new_node, old_node in enumerate(sorted_nodes)}
    G = nx.relabel_nodes(G, mapping)


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


def infer_width(G: nx.graph) -> EdgeDict3:
    distr = {
        "primary": {
            True: [872, 2757, 934, 382, 95, 47, 1, 4, 0, 22],
            False: [0, 926, 296, 1894, 220, 280, 58, 64, 2, 0],
        },
        "residential": {
            True: [2227, 405, 19, 4, 0, 0, 0, 0, 0, 0],
            False: [220, 1338, 74, 30, 0, 0, 0, 0, 0, 0],
        },
        "living_street": {
            True: [152, 4, 2, 0, 0, 0, 0, 0, 0, 0],
            False: [20, 24, 2, 0, 0, 0, 0, 0, 0, 0],
        },
        "motorway": {
            True: [6, 135, 64, 80, 8, 0, 0, 0, 0, 0],
            False: [6, 135, 64, 80, 8, 0, 0, 0, 0, 0],
        },
        "tertiary": {
            True: [768, 794, 140, 15, 3, 1, 0, 0, 0, 0],
            False: [14, 3200, 382, 168, 6, 12, 0, 0, 0, 0],
        },
        "trunk_link": {
            True: [326, 481, 48, 0, 0, 0, 0, 0, 0, 0],
            False: [326, 481, 48, 0, 0, 0, 0, 0, 0, 0],
        },
        "motorway_link": {
            True: [63, 355, 15, 2, 0, 0, 0, 0, 0, 0],
            False: [63, 355, 15, 2, 0, 0, 0, 0, 0, 0],
        },
        "secondary": {
            True: [629, 1432, 452, 128, 29, 26, 0, 2, 0, 0],
            False: [32, 2706, 880, 606, 40, 46, 0, 0, 0, 0],
        },
        "primary_link": {
            True: [162, 125, 22, 4, 2, 0, 0, 0, 0, 0],
            False: [0, 4, 4, 0, 0, 0, 0, 0, 0, 0],
        },
        "tertiary_link": {
            True: [17, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            False: [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        },
        "unclassified": {
            True: [296, 72, 19, 5, 4, 0, 3, 0, 0, 0],
            False: [12, 288, 28, 16, 0, 0, 0, 0, 0, 0],
        },
        "secondary_link": {
            True: [34, 39, 7, 1, 0, 0, 0, 0, 0, 0],
            False: [34, 39, 7, 1, 0, 0, 0, 0, 0, 0],
        },
        "trunk": {
            True: [0, 94, 185, 968, 30, 2, 0, 0, 0, 0],
            False: [0, 94, 185, 968, 30, 2, 0, 0, 0, 0],
        },
        "crossing": {
            True: None,
            False: None,
        },
        "emergency_access_point": {
            True: None,
            False: None,
        },
        "disused": {
            True: None,
            False: None,
        },
    }
    widths, lanes = {}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 12]
    existing_widths = nx.get_edge_attributes(G, "width")
    existing_lanes = nx.get_edge_attributes(G, "lanes")
    highways = nx.get_edge_attributes(G, "highway")
    oneways = nx.get_edge_attributes(G, "oneway")
    for u, v, w in G.edges:
        try:
            widths[(u, v, w)] = int(existing_widths[(u, v, w)])
        except:
            try:
                widths[(u, v, w)] = int(existing_lanes[(u, v, w)]) * 4
                # 4 étant la largeur moyenne d'une rue parisienne
            except:
                widths[(u, v, w)] = (
                    rd.choices(
                        lanes, weights=distr[highways[(u, v, w)]][oneways[(u, v, w)]]
                    )[0]
                    * 4
                )
    return widths


def infer_lanes(G: nx.graph) -> EdgeDict3:
    widths = {}
    existing_widths = nx.get_edge_attributes(G, "width")
    existing_lanes = nx.get_edge_attributes(G, "lanes")
    highways = nx.get_edge_attributes(G, "highway")
    for u, v, w in G.edges:
        try:  # 4 étant la largeur moyenne d'une rue parisienne
            widths[(u, v, w)] = math.ceil(existing_widths[(u, v, w)] / 4)
        except:
            if highways[(u, v, w)] == "primary" or highways[(u, v, w)] == "secondary":
                try:
                    val = int(existing_lanes[(u, v, w)])
                    if val < 3:
                        widths[(u, v, w)] = 3
                    else:
                        widths[(u, v, w)] = val
                except:
                    widths[(u, v, w)] = 3
            else:
                try:
                    widths[(u, v, w)] = int(existing_lanes[(u, v, w)])
                except:
                    widths[(u, v, w)] = 2
    return widths

def propagate_bridges(G: nx.Graph, bridge_dict: dict, k: int = 10) -> None:
    """Changes bridge_dict in place for propagating each bridge to the k nearest neighbors"""
    def rec_propagate_bridges(edge: Edge, G: nx.Graph, seen: list[Edge], bridge_dict: dict, k: int):
        if edge in bridge_dict:
            bridge_dict[edge] = "yes"
            seen.append(edge)
        else: # case (n2, n1) is not an edge (but (n1, n2) is)
            return
        if k == 0:
            return
        for neighbor1 in G.neighbors(edge[0]):
            rec_propagate_bridges((edge[0], neighbor1), G, seen, bridge_dict, k-1)
            rec_propagate_bridges((neighbor1, edge[0]), G, seen, bridge_dict, k-1)
        for neighbor2 in G.neighbors(edge[1]):
            rec_propagate_bridges((edge[1], neighbor2), G, seen, bridge_dict, k-1)
            rec_propagate_bridges((neighbor2, edge[1]), G, seen, bridge_dict, k-1)

    seen_edges = []
    for edge, is_bridge in bridge_dict.items():
        if is_bridge == "yes":
            rec_propagate_bridges(edge, G, seen_edges, bridge_dict, k)

def preprocessing(
    G: nx.Graph,
    cost_name: str,
    minmax: tuple[int, int] | None,
    distrib: dict[int, float] | None,
):
    """
    Does all the required preprocessing in place

    @params:
        Required: G the networkx graph
        Required: cost_name the name of the cost function, available names are
            - width
            - squared width
            - width with maxspeed
            - width without tunnel
            - width without bridge
            - random(min, max)
            - random distribution
            - lanes
            - squared lanes"
            - lanes with maxspeed
            - lanes wihtout bridge 
            - betweenness /!\ the graph must have betweenness as attribute for each edges /!\ 
            - _ will be considered as weight 1 everywhere
        Optional: minmax a tuple of min and max values for cost random(min, max)
        Optional: distribution a distribution of frequencies to respect for cost random distribution

    @returns:
        None
    """
    inf: int = 95099713  # big number for removing cut access to an edge
    if cost_name in [
        "width",
        "squared width",
        "width without tunnel",
        "width without bridge",
        "width with maxspeed",
    ]:
        edge_width = infer_width(G)
    elif cost_name in [
        "lanes",
        "squared lanes",
        "lanes with maxspeed",
        "lanes without bridge",
    ]:
        edge_lanes = infer_lanes(G)
    match cost_name:
        case "width":
            edge_weight = edge_width
        case "squared width":
            edge_weight = {k: v**2 for k, v in edge_width.items()}
        case "width with maxspeed":
            maxspeed_dict = nx.get_edge_attributes(G, "maxspeed", default=50)
            edge_weight = {
                k: (
                    v
                    if maxspeed_dict[k] == "walk" or int(maxspeed_dict[k]) <= 50
                    else inf
                )
                for k, v in edge_width.items()
            }
        case "width without bridge":
            bridge_dict = nx.get_edge_attributes(G, "bridge", default="no")
            propagate_bridges(G, bridge_dict)
            edge_weight = {
                k: inf if bridge_dict[k] == "yes" else v for k, v in edge_width.items()
            }
        case "width without tunnel":
            tunnel_dict = nx.get_edge_attributes(G, "tunnel", default=False)
            edge_weight = {
                k: v if not tunnel_dict[k] else inf for k, v in edge_width.items()
            }
        case "random(min, max)":
            if type(minmax) is None:
                raise TypeError(
                    "argument minmax should be given if minmax cost is selected"
                )
            edge_weight = {k: rd.randint(minmax[0], minmax[1]) for k in G.edges}
        case "random distribution":
            if type(distrib) is None:
                raise TypeError(
                    "argument distrib should be given if distrib cost is selected"
                )
            edge_weight = {
                k: rd.choices(list(distrib.keys()), weights=list(distrib.values()))[0]
                for k in G.edges
            }
        case "lanes":
            edge_weight = edge_lanes
        case "squared lanes":
            edge_weight = {k: v**2 for k, v in edge_lanes.items()}
        case "lanes with maxspeed":
            maxspeed_dict = nx.get_edge_attributes(G, "maxspeed", default=50)
            edge_weight = {
                k: (
                    v
                    if maxspeed_dict[k] == "walk" or int(maxspeed_dict[k]) <= 50
                    else inf
                )
                for k, v in edge_lanes.items()
            }
        case "lanes without bridge":
            bridge_dict = nx.get_edge_attributes(G, "bridge", default=False)
            edge_weight = {
                k: inf if bridge_dict[k] == "yes" else v for k, v in edge_lanes.items()
            }
        case "betweenness":
            edge_weight = nx.get_edge_attributes(G, "betweenness")
        case _:
            edge_weight = {k: 1 for k in G.edges}
    nx.set_edge_attributes(G, edge_weight, "weight")
    G.remove_edges_from(nx.selfloop_edges(G))
    add_node_weights_and_relabel(G)
    replace_parallel_edges(G)
    G.to_undirected()


def init_city_graph(filepath, betweenness: bool = False):
    # create, project, and consolidate a graph
    G = ox.graph_from_place(
        "Paris, Paris, France",
        network_type="drive",
        buffer_dist=350,
        simplify=False,
        retain_all=True,
        clean_periphery=False,
        truncate_by_edge=False,
    )
    G_Paris = ox.project_graph(
        G, to_crs="epsg:2154"
    )  ## pour le mettre dans le même référentiel que les données de Paris

    print("Just after importation, we have : ")
    print(str(len(G.edges())) + " edges")
    print(str(len(G.nodes())) + " nodes")
    G2 = ox.consolidate_intersections(
        G_Paris, rebuild_graph=True, tolerance=4, dead_ends=True
    )
    print("After consolidation, we have : ")
    print(str(len(G2.edges())) + " edges")
    print(str(len(G2.nodes())) + " nodes")
    G_out = ox.project_graph(G2, to_crs="epsg:4326")
    print("After projection, we have : ")
    print(str(len(G_out.edges())) + " edges")
    print(str(len(G_out.nodes())) + " nodes")

    if betweenness:
        bc = nx.edge_betweenness_centrality(G_out)
        nx.set_edge_attributes(G_out, bc, "betweenness")
    ox.save_graphml(G_out, filepath=filepath)


# init_city_graph("./data/Paris.graphml")


def prepare_instance(
    read_filename: str,
    write_filename: str,
    val_name: str,
    minmax: tuple[int, int] | None = None,
    distr: dict[int, float] | None = None,
):
    """ "
    Prepare a json KaHIP Graph instance according to the required cost function.

    Cost options:
        - "no val"
        - "width"
        - "squared width"
        - "width with maxspeed"
        - "width without bridge"
        - "width without tunnel"
        - "random(min, max)"
        - "random distribution"
        - "lanes"
        - "squared lanes"
        - "lanes with maxspeed"
        - "lanes without bridge"
    """
    print("Loading instance")
    G_nx = ox.load_graphml(read_filename)
    print(f"preprocessing the graph...")
    preprocessing(G_nx, val_name, minmax, distr)
    print("Conversion into KaHIP format...")
    G_kp = Graph(nx=G_nx)
    G_kp.save_graph(write_filename)


def flatten(l):
    if isinstance(l, str):
        return [l]
    res = []
    for x in l:
        res += flatten(x)
    return res


def thousand_cuts(kp_paths: list[str], costs_names: list[str], imbalances: list[float]):
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
                G_kp.kaffpa_cut(2, imbalance, 0, seed, 2)
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


def clustering_procedure():
    print("import stuff...")
    G_nx = ox.load_graphml(graphml_path[2])
    G_kp = Graph(json=kp_paths[11])
    with open(cut_paths_2[2], "r") as read_file:
        kcuts = json.load(read_file)
    cuts = {}
    for k, (edgecut, blocks) in kcuts.items():
        G_kp.set_last_results(edgecut, blocks)
        cuts[k] = G_kp.process_cut()

    print("clustering...")
    C = CutsClassification(cuts, G_nx)
    n = 7500
    C.cluster_louvain("sum", n)
    print(f"for n = {n}")
    for level in C._levels:
        print(len(level))
    C.save_last_classes("data/clusters/CTS_" + str(n) + "_lanesmaxspeed.json")


def clustering_display():
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
