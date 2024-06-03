import networkx as nx
import osmnx as ox
import random as rd
import json
import math

from Graph import Graph
from typ import EdgeDict3


def replace_parallel_edges(G: nx.Graph) -> None:
    """
    KaHIP doesn't support parallel edges, so they're replaced by an intermediary node for a new edge.
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

        if edges_weight:
            edges_weight[(u, new_node, 0)] = int(edges_weight[(u, v, k)])
            edges_weight[(new_node, v, 0)] = int(edges_weight[(u, v, k)])

        G.edges[(u, v, 0)].update(G.edges[(u, v, k)])
        G.edges[(u, new_node, 0)].update(G.edges[(u, v, 0)])
        G.edges[(new_node, v, 0)].update(G.edges[(u, v, 0)])

        G.edges[(u, new_node, 0)]["length"] = G.edges[(u, v, 0)]["length"] / 2
        G.edges[(new_node, v, 0)]["length"] = G.edges[(u, v, 0)]["length"] / 2

        G.remove_edge(u, v, k)

    node_weights = {}
    for node in G.nodes:
        node_weights[node] = 1

    nx.set_node_attributes(G, node_weights, "weight")
    nx.set_edge_attributes(G, edges_weight, "weight")


def add_node_weights_and_relabel(G: nx.Graph) -> None:
    """
    The relabeling consist in changing node index such that the first node has the index 0 and the last the index n-1
    The objective is to create a correspondance with the adjacency lists structure.
    """
    w_nodes = {}
    for node in list(G.nodes):
        w_nodes[node] = 1
    nx.set_node_attributes(G, w_nodes, "weight")
    sorted_nodes = sorted(G.nodes())
    mapping = {old_node: new_node for new_node, old_node in enumerate(sorted_nodes)}
    G = nx.relabel_nodes(G, mapping)


def infer_lanes(G: nx.graph) -> EdgeDict3:
    """Returns the approximate number of lanes for each edge."""
    widths = {}
    existing_widths = nx.get_edge_attributes(G, "width")
    existing_lanes = nx.get_edge_attributes(G, "lanes")
    highways = nx.get_edge_attributes(G, "highway")
    for u, v, w in G.edges:
        try:  # 4 meters being the average width of a Paris street
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


def propagate_bridges(G: nx.Graph, bridge_dict: dict, neighborhood_fp: str) -> dict:
    """Returns a new bridge_dict with bridges propagated to its geographical neighbors and removes the periph as a bridge"""
    with open(neighborhood_fp, "r") as neighbors_file:
        data = json.load(neighbors_file)
    neighborhood = {}
    for k, v in data.items():
        neighborhood[eval(k)] = [eval(e) for e in v]
    res_dict = {}
    highways = nx.get_edge_attributes(G, "highway")  # pour enlever le périph
    for (u, v, w), is_bridge in bridge_dict.items():
        not_periph = not highways[(u, v, w)] in [
            "trunk",
            "motorway",
            "motorway_link",
            "trunk_link",
        ]
        if is_bridge == "yes" and not_periph:
            try:
                for a, b in neighborhood[(u, v)]:
                    res_dict[(a, b, 0)] = "yes"
            except:
                for a, b in neighborhood[(v, u)]:
                    res_dict[(a, b, 0)] = "yes"
        else:
            res_dict[(u, v, 0)] = "no"
    return res_dict


def preprocessing(
    G: nx.Graph,
    cost_name: str,
    minmax: tuple[int, int] | None,
    distrib: dict[int, float] | None,
    neighbor_fp: str | None,
):
    """
    Does all the required preprocessing in place

    @params:
        Required: G the networkx graph
        Required: cost_name the name of the cost function, available names are
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
        "lanes",
        "squared lanes",
        "lanes with maxspeed",
        "lanes without bridge",
    ]:
        edge_lanes = infer_lanes(G)
    match cost_name:
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
            if not neighbor_fp:
                raise ValueError(
                    "The neighborhood file must be given for 'without bridge' computations"
                )
            bridge_dict = nx.get_edge_attributes(G, "bridge", default="no")
            new_bridge_dict = propagate_bridges(G, bridge_dict, neighbor_fp)
            nx.set_edge_attributes(G, new_bridge_dict, "bridge")
            edge_weight = {
                k: inf if new_bridge_dict[k] == "yes" else v
                for k, v in edge_lanes.items()
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


def init_city_graph(filepath, betweenness: bool = False, city_name: str = "Paris"):
    """Instantiate a city graph using OSMnx library."""
    match city_name:
        case "Paris":
            city = "Paris, Paris, France"
            buffer = 350
            epsg = "epsg:2154"
            tol = 4
        case "Shanghai":
            city = "Shanghai, China"
            epsg = "epsg:2335"
            tol = 4  # tester plus grand
            buffer = 500
        case "Manhattan":
            city = "Manhattan, New York, USA"
            tol = 4  # tester plus grand
            epsg = "epsg:26918"
            buffer = 500
    # create, project, and consolidate a graph
    G = ox.graph_from_place(
        city,
        network_type="drive",
        buffer_dist=buffer,
        simplify=False,
        retain_all=True,
        clean_periphery=False,
        truncate_by_edge=False,
    )
    G_Projected = ox.project_graph(
        G, to_crs=epsg
    )  ## pour le mettre dans le même référentiel que les données de Paris

    print("Just after importation, we have : ")
    print(str(len(G.edges())) + " edges")
    print(str(len(G.nodes())) + " nodes")
    G2 = ox.consolidate_intersections(
        G_Projected, rebuild_graph=True, tolerance=tol, dead_ends=True
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


def prepare_instance(
    read_filename: str,
    write_filename: str,
    val_name: str,
    minmax: tuple[int, int] | None = None,
    distr: dict[int, float] | None = None,
    fp_neighbors: str | None = None,
):
    """ "
    Prepare a json KaHIP Graph instance according to the required cost function.

    Cost options:
        - "no val"
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
    preprocessing(G_nx, val_name, minmax, distr, fp_neighbors)
    print("Conversion into KaHIP format...")
    G_kp = Graph(nx=G_nx)
    G_kp.save_graph(write_filename)
