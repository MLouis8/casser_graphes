# from cdlib.algorithms import walktrap
import networkx as nx
import numpy as np
import json

from typ import Edge

def aggregate_graph(G_base: nx.Graph, node_dict: dict):
    G = nx.Graph()
    G.add_nodes_from(set(node_dict.values()))
    for edge in G_base.edges:
        n1, n2 = node_dict[edge[0]], node_dict[edge[1]]
        if node_dict[edge[0]] != node_dict[edge[1]] and not G.has_edge(n1, n2):
            G.add_edge(n1, n2)
    return G

def walktrap_communities_wrapper(G_nx: nx.Graph, fp: str, k: int = 1) -> None:
    """Repeats the walktrap algorithm k times to obtain bigger clusters"""
    node_community_dict = { node: node for node in G_nx.nodes }
    G = G_nx.copy()
    for i in range(k):
        print(f"{i}th level")
        comms = None #walktrap(G)
        for node in G_nx.nodes:
            for j, com in enumerate(comms.communities):
                if node in com:
                    break
            node_community_dict[node] = j
        print(len(comms.communities))
        G = aggregate_graph(G_nx, node_community_dict)
    result = np.empty(max(list(node_community_dict.values()))+1, dtype=list)
    for node, com in node_community_dict.items():
        if result[com]:
            result[com].append(node)
        else:
            result[com] = [node]
    with open(fp, "w") as wfile:
        json.dump(list(result), wfile)

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
        if (n1 != None) and (n2 != None) and (n1 != n2):
            cut_edges.append(e)
    return cut_edges