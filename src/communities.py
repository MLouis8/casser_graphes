from cdlib.algorithms import walktrap
import networkx as nx
import json

from typ import Edge

def walktrap_communities_wrapper(G_nx: nx.Graph, fp: str, k: int = 1) -> None:
    comms = walktrap(G_nx)
    for i in range(k-1):
        G = nx.Graph()
        for com in comms.communities:
            pass
    with open(fp, "w") as wfile:
        json.dump(comms.communities, wfile)

def louvain_communities_wrapper(G_nx: nx.Graph, fp: str, res: float = 0.004) -> None:
    weights = nx.get_edge_attributes(G_nx, 'weight')
    new_weight = {}
    for k, v in weights.items():
        new_weight[k] = eval(v)
    nx.set_edge_attributes(G_nx, new_weight, 'weight')
    comms = nx.community.louvain_communities(G_nx, weight='weight', resolution=res)
    res = []
    with open(fp, "w") as wfile:
        json.dump([list(com) for com in comms], wfile)

def determine_cut_edges(G_nx: nx.Graph, parts: list[int]) -> list[Edge]:
    cut_edges = []
    for e in G_nx.edges:
        n1, n2 = None, None
        for i, part in enumerate(parts):
            if e[0] in part:
                n1 = i
            if e[1] in part:
                n2 = i
            if n1 and n2:
                break
        if n1 != n2:
            cut_edges.append(e)
    return cut_edges