import networkx as nx
import osmnx as ox
import json

from typ import EdgeCoord


def dist(e1: EdgeCoord, e2: EdgeCoord) -> float:
    return min([
        ox.distance.great_circle(e1[0][0], e1[0][1], e2[0][0], e2[0][1]),
        ox.distance.great_circle(e1[0][0], e1[0][1], e2[1][0], e2[1][1]),
        ox.distance.great_circle(e1[1][0], e1[1][1], e2[0][0], e2[0][1]),
        ox.distance.great_circle(e1[1][0], e1[1][1], e2[1][0], e2[1][1])
    ])


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
    within_range = lambda x, y: ox.distance.great_circle(xs[x], ys[x], xs[y], ys[y]) < k
    for a, b, _ in G_nx.edges:
        if str((a, b)) in neighborhoods.keys():
            continue
        neighborhoods[str((a, b))] = []
        for c, d, _ in G_nx.edges:
            if (
                within_range(a, c)
                or within_range(a, d)
                or within_range(b, c)
                or within_range(b, d)
            ):
                neighborhoods[str((a, b))].append(str((c, d)))
    with open(fp, "w") as save_file:
        json.dump(neighborhoods, save_file)
