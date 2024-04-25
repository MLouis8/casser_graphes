import networkx as nx
import osmnx as ox
import json
import math

from typ import Edge, Coord

def euclidian_dist(e1: Coord, e2: Coord) -> float:
    return math.dist(e1, e2)

def neighborhood_procedure(G_nx: nx.Graph, k: int, fp: str) -> None:
    """
    This procedure computes all neighbors of each edge of a osmnx Graph containing x, y coordinates for nodes.

    Parameters:
        G_nx: nx.Graph, the networkx graph coming from osmnx (must have x, y attributes for each nodes)
        k: int, neighborhood (in meters)
        fp: str, filepath for saving (must be json format)

    Saves:
        It saves a json dict mapping each edge to the list of edges in its neighborhood
    """
    neighborhoods = {}
    xs = nx.get_node_attributes(G_nx, "x")
    ys = nx.get_node_attributes(G_nx, "y")
    edge_coord = lambda n1, n2: (xs[n1] + xs[n2] / 2, ys[n1] + ys[n2] / 2)
    for n1, n2, _ in G_nx.edges:
        if str((n2, n1)) in neighborhoods.keys():
            continue
        neighborhoods[str((n1, n2))] = []
        lat1, long1 = edge_coord(n1, n2)
        for possible_neighbor in G_nx.edges:
            lat2, long2 = edge_coord(possible_neighbor[0], possible_neighbor[1])
            if ox.distance.great_circle(lat1, long1, lat2, long2) < k:
                neighborhoods[str((n1, n2))].append(str((possible_neighbor[0], possible_neighbor[1])))
    with open(fp, "w") as save_file:
        json.dump(neighborhoods, save_file)
