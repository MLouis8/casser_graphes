from Graph import Graph
from typ import RobustList, Cuts, Edge, EdgeDict

import json
import random as rd
import networkx as nx
import numpy as np
from typing import Any
from copy import deepcopy


def freq_attack(G: Graph, ncuts: int) -> Edge:
    cut_union = []
    seen_seeds = []
    for ncut in range(ncuts):
        seed = rd.randint(0, 1044642763)
        while seed in seen_seeds:
            seed = rd.randint(0, 1044642763)
        seen_seeds.append(seed)
        G.kaffpa_cut(2, 0.05, 0, seed, 3)
        cut_union += G.process_cut()

    frequencies = {}
    for edge in cut_union:
        if edge in frequencies:
            frequencies[edge] += 1
        else:
            frequencies[edge] = 1
    return max(frequencies, key=frequencies.get)


def betweenness_attack(G: Graph) -> Edge:
    bc = G.get_edge_bc()
    return max(bc, key=bc.get)


def random_attack(G: Graph, n: int) -> list[Edge]:
    return rd.choices(list(G._nx.edges), k=n)


def maxdegree_attack(G: Graph) -> Edge:
    maxdegree, chosen_edge = 0, None
    for edge in G._nx.edges:
        degree = G._nx.degree[edge[0]] * G._nx.degree[edge[1]]
        maxdegree = maxdegree if maxdegree > degree else degree
        chosen_edge = edge
    return chosen_edge


def attack(
    G: Graph,
    k: int,
    fp_save: str,
    order: str,
    metric_bc: bool,
    metric_cc: bool,
    ncuts: int = 1000,
    nrandoms: int = 100,
    save: bool = True,
    # extended: bool = False, TODO: don't recalculate first bc when extended
) -> RobustList | None:
    """
    Simulates an attack on a Graph with the following strategy:
        repeat k times:
        - cuts ncuts times G
        - remove the most cut edge

    Parameters:
        G: Graph
        k: int, the number of edges to remove
        fp_save: str, the saving path
        order: str, can be "bc", "freq", "deg" or "rd" depending on the strategy to apply for the attack
            (ex. bc will remove the highest Betweenness Centrality edge)
        metric_bc: bool, whether the Betweenness Centrality is computed at each step
        metric_cc: bool, whether the max size of the connected components is computed or not
        ncuts: int (default 1000), the number of cuts to decide
        nrandoms: int (default 100), the number of random edges to choose for the mean

    Saves:
        List of size k, containing a tuple of size 3 containing:
         - removed edge
         - 0, 1 or 2 metrics applied to the graph at this step
    """

    def not_rd_procedure(metrics, chosen_edge):
        bc = G.get_edge_bc(new=True) if metric_bc else None
        cc = G.get_size_biggest_cc if metric_cc else None
        metrics.append((chosen_edge, bc, cc))

    def rd_procedure(metrics, chosen_edges):
        if len(chosen_edges) > 0:
            bcs, cc_list = [], []
            for edge in chosen_edges:
                G_copy = deepcopy(G)
                G_copy.remove_edge(edge)
                bcs.append(np.mean(list(G_copy.get_edge_bc(new=True).values())))
                cc_list.append(G_copy.get_size_biggest_cc)
            metrics.append((chosen_edges, bcs, cc_list))
        else:
            metrics.append(
                (
                    None,
                    np.mean(list(G.get_edge_bc(new=True).values())),
                    G.get_size_biggest_cc,
                )
            )

    metrics, chosen_edge, chosen_edges = [], None, []
    for i in range(k):
        print(f"processing the {i+1}-th attack over {k}, order: {order}")
        match order:
            case "bc":
                not_rd_procedure(metrics, chosen_edge)
                chosen_edge = betweenness_attack(G)
                G.remove_edge(chosen_edge)
            case "freq":
                not_rd_procedure(metrics, chosen_edge)
                chosen_edge = freq_attack(G, ncuts)
                G.remove_edge(chosen_edge)
            case "deg":
                not_rd_procedure(metrics, chosen_edge)
                chosen_edge = maxdegree_attack(G)
                G.remove_edge(chosen_edge)
            case "rd":
                rd_procedure(metrics, chosen_edges)
                chosen_edges = random_attack(G, nrandoms)
    if order != "rd":
        not_rd_procedure(metrics, chosen_edge)
    else:
        rd_procedure(metrics, chosen_edges)
    if save:
        if order != "rd":
            temp = metrics.copy()
            metrics = []
            for step in temp:
                str_d = {str(k): v for k, v in step[1].items()}
                metrics.append([str(step[0]), str_d, step[2]])
        else:
            temp = metrics.copy()
            metrics = []
            for step in temp:
                edges = [str(e) for e in step[0]] if step[0] else None
                metrics.append([edges, step[1], step[2]])
        with open(fp_save, "w") as save_file:
            json.dump(metrics, save_file)
    else:
        return metrics


def avg_bc_edge_subset(G: nx.Graph, s: list[Edge]):
    """
    Returns the average BC of the all graph compared to the average of the listed edges,
    the graph must have the attribute betweenness
    """
    avg1, cpt1 = 0, 0
    avg2, cpt2 = 0, 0
    for edge in G.edges(data=True):
        avg1 += edge[2]["betweenness"]
        cpt1 += 1
        if (edge[0], edge[1]) in s or (edge[1], edge[0]) in s:
            avg2 += edge[2]["betweenness"]
            cpt2 += 1
    return avg1 / cpt1, avg2 / cpt2


def extend_attack(
    G: Graph,
    metrics: RobustList,
    k: int,
    fp_save: str,
    order: str,
    metric_bc: bool,
    metric_cc: bool,
    ncuts: int,
    nrandoms: int,
    save: bool,
) -> RobustList | None:
    # remove the already processed edges
    for edge, _, _ in metrics:
        G.remove_edge(edge)
    # launch attack on the new graph
    tail = attack(
        G,
        k,
        fp_save,
        order,
        metric_bc,
        metric_cc,
        ncuts=ncuts,
        nrandoms=nrandoms,
        save=False,
        #  extended=True, see attack for more details
    )
    # return the concat of the two metrics
    if save:
        with open(fp_save, "w") as saving_file:
            json.dump(metrics + tail, saving_file)
    else:
        return metrics + tail
