from Graph import Graph
from typ import RobustList, Edge, EdgeDict
from geo import dist

import json
import random as rd
import networkx as nx
import numpy as np
from copy import deepcopy

def freq_attack(G: Graph, ncuts: int) -> Edge:
    cut_union = []
    seen_seeds = []
    for _ in range(ncuts):
        seed = rd.randint(0, 1044642763)
        while seed in seen_seeds:
            seed = rd.randint(0, 1044642763)
        seen_seeds.append(seed)
        G.kaffpa_cut(2, 0.05, 0, seed, 2)
        cut_union += G.process_cut()

    frequencies = {}
    for edge in cut_union:
        if edge in frequencies:
            frequencies[edge] += 1
        else:
            frequencies[edge] = 1
    try:
        return max(frequencies, key=frequencies.get)
    except:
        return None

def betweenness_attack(G: Graph, weighted: bool, subset: list[Edge] | None) -> Edge:
    bc = G.get_edge_bc(weighted=weighted)
    if subset:
        res, max_bc = None, 0
        for edge in subset:
            if edge in bc and bc[edge] > max_bc:
                max_bc = bc[edge]
                res = edge
        return res
    else:
        return max(bc, key=bc.get)


def random_attack(G: Graph, n: int, subset: list[Edge] | None) -> list[Edge]:
    if subset:
        return rd.choice(subset)
    return rd.choices(list(G._nx.edges), k=n)


def maxdegree_attack(G: Graph, subset: list[Edge] | None) -> Edge:
    maxdegree, chosen_edge = 0, None
    for edge in G._nx.edges:
        degree = G._nx.degree[edge[0]] * G._nx.degree[edge[1]]
        maxdegree = maxdegree if maxdegree > degree else degree
        chosen_edge = edge
    return chosen_edge


def best_pertubation(G: Graph, pertubation: tuple[str, float], subset: list[Edge] | None) -> Edge:
    """
    Try to remove each edge, computing the resulting pertubation.
    Returns the edge with the highest perturbation score

    Available perturbations:
        - ('dmax', treshold)
        - ('nimpacts', treshold)
        - ('sumdiffs', treshold)
        - ('maxdiff', treshold)
        # - ('sumbydist', treshold)
        # - ('sumbyinvdist', treshold)
    See measure_bc_impact for more information about perturbations.
    """
    
    measure_bc_impact()


# def cascading_attack(G: Graph, treshold):

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
    subset: list[Edge] | None = None,
    weighted: bool = True
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
        save: bool, whether the result should be returned or saved (used for extended save)
        subset: list[Edge] | None, if set, the edges will be chosen in the subset and the metrics are still computed for the entire graph

        Warning subset and frequency attack aren't available at the same time.
    Saves:
        List of size k, containing a tuple of size 3 containing:
         - removed edge
         - 0, 1 or 2 metrics applied to the graph at this step
    """

    def not_rd_procedure(metrics, chosen_edge):
        bc = G.get_edge_bc(weighted=weighted, new=True) if metric_bc else None
        cc = G.get_size_biggest_cc if metric_cc else None
        metrics.append((chosen_edge, bc, cc))

    def rd_procedure(metrics, chosen_edges):
        if len(chosen_edges) > 1:
            bcs, cc_list = [], []
            for edge in chosen_edges:
                G_copy = deepcopy(G)
                G_copy.remove_edge(edge)
                if metric_bc:
                    bcs.append(np.mean(list(G_copy.get_edge_bc(weighted=weighted, new=True).values())))
                if metric_cc:
                    cc_list.append(G_copy.get_size_biggest_cc if metric_cc else None)
            metrics.append((chosen_edges, bcs, cc_list))
        elif len(chosen_edges) == 1 or len(chosen_edges) == 0:
            not_rd_procedure(metrics, chosen_edges[0] if len(chosen_edges) == 1 else None)
        else:
            metrics.append(
                (
                    None,
                    np.mean(list(G.get_edge_bc(weighted=weighted, new=True).values())) if metric_bc else None,
                    G.get_size_biggest_cc if metric_cc else None,
                )
            )
    if order == "freq" and subset:
        raise ValueError("Freq attack not available when considering only a subset of edges")
    metrics, chosen_edge, chosen_edges, direct_save = [], None, [], False
    for i in range(k):
        print(f"processing the {i+1}-th attack over {k}, order: {order}")
        match order:
            case "bc":
                not_rd_procedure(metrics, chosen_edge)
                chosen_edge = betweenness_attack(G, weighted, subset)
                G.remove_edge(chosen_edge)
            case "freq":
                not_rd_procedure(metrics, chosen_edge)
                chosen_edge = freq_attack(G, ncuts)
                if not chosen_edge:
                    direct_save = True
                    break
                G.remove_edge(chosen_edge)
            case "deg":
                not_rd_procedure(metrics, chosen_edge)
                chosen_edge = maxdegree_attack(G, subset)
                G.remove_edge(chosen_edge)
            case "rd":
                rd_procedure(metrics, chosen_edges)
                G._nx = G.to_nx()
                chosen_edges = random_attack(G, nrandoms, subset)
    if order != "rd" and not direct_save:
        not_rd_procedure(metrics, chosen_edge)
    elif not direct_save:
        rd_procedure(metrics, chosen_edges)
    if save:
        if order != "rd" or nrandoms == 1:
            temp = metrics.copy()
            metrics = []
            for step in temp:
                edges = [str(e) for e in step[0]] if step[0] else None
                str_d = {str(k): v for k, v in step[1].items()} if step[1] else None
                cc = step[2] if step[2] else None
                metrics.append([edges, str_d, cc])
        else:
            raise ValueError("not done yet")
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
    subset: list[Edge] | None = None,
    weighted: bool = False
) -> RobustList | None:
    # remove the already processed edges
    for e, _, _ in metrics:
        try:
            edge = eval(e)
            if edge is None:
                continue
        except:
            if e is None:
                continue
            edge = (eval(e[0]), eval(e[1]))
        G.remove_edge(edge)
        if subset:
            subset.remove(edge)
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
        subset=subset,
        weighted=weighted
        #  extended=True, see attack for more details
    )
    # return the concat of the two metrics
    if order != "rd":
        temp = tail.copy()
        tail = []
        for step in temp:
            str_d = {str(k): v for k, v in step[1].items()}
            tail.append([str(step[0]), str_d, step[2]])
    else:
        temp = tail.copy()
        tail = []
        for step in temp:
            edges = [str(e) for e in step[0]] if step[0] else None
            tail.append([edges, step[1], step[2]])
    if save:
        with open(fp_save, "w") as saving_file:
            json.dump(metrics + tail, saving_file)
    else:
        return metrics + tail

def verify_integrity(fp: str, order: str, size: int) -> None:
    """"
    Verifies intergrity of the robustness list
    
    Warning: Needs a complete robustness list (cc and bc saved)

    Checks:
    - size of the list
    - that the removed edges are the good one (for bc)
    - that the metrics (edge Betweenness Centrality and biggest Connected Component) are present and well-formed

    Parameters:
        fp: str, the filepath where the attack is stored
        order: str, can be "bc", "freq", "deg" or "rd" is the criterion chosen for removing edges
        size: int, the number of removed edges
    
    Returns nothing but raises exceptions when data isn't valid
    """
    with open(fp, "r") as read_file:
        data = json.load(read_file)
    # checking size
    assert len(data) == size + 1
    # verifying removed edges
    removed_edges = []
    for i, attack in enumerate(data):
        edge, bc, cc = eval(attack[0]), attack[1], attack[2]
        if i > 0:
            if isinstance(edge, tuple) and isinstance(edge[0], int) and isinstance(edge[1], int):
                removed_edges.append(edge)
            else:
                raise TypeError(f"First element of the attack tuple must be the removed edge. Instead type {type(edge)} found")
        else:
            assert edge == None
        if type(bc) != dict:
            raise TypeError(f"the betweenness centrality measures should be of type dict instead {type(bc)} is found")
        for k, v in bc.items():
            if not isinstance(k, str) or not isinstance(v, float):
                raise TypeError(f"the betweenness centrality measures should be of type dict[str, float] instead {type(bc)}[{type(k)}, {type(v)}] is found")
        if type(cc) != int:
            raise TypeError(f"the biggest connected component measures should be of type int, instead {type(cc)} is found")
    for j, edge in enumerate(removed_edges):
        if order == "bc":
            assert eval(max(data[j][1], key=data[j][1].get)) == edge
        assert not edge in data[j+1][1]

def measure_strong_connectivity(robust_list: RobustList, G_nx: nx.Graph) -> list[list[int]]:
    """Takes a robust dict and a graph and returns for each step the size of every connected component"""
    res = []
    G = G_nx.copy()
    for attack in robust_list:
        if not attack[0]:
            continue
        try:
            edge = (eval(attack[0][0]), eval(attack[0][1]))
        except:
            edge = eval(attack[0])
        if edge:
            try:
                G.remove_edge(edge[0], edge[1])
            except:
                G.remove_edge(edge[1], edge[0])
        res.append(len(max(nx.strongly_connected_components(G), key=len)))
    return res

def measure_diameter(robust_list: RobustList, G_nx: nx.Graph) -> list[int]:
    """Takes a robust dict and a graph and returns for each step the diameter of the graph"""
    res = []
    for attack in robust_list:
        if attack[0] is None:
           res.append(nx.diameter(G_nx))
        else:
            try:
                e = eval(attack[0])
                G_nx.remove_edge(e[0], e[1])
            except:
                G_nx.remove_edge(eval(attack[0][0]), eval(attack[0][1]))
            res.append(nx.diameter(G_nx))
    return res

def measure_bc_impact(bc1: EdgeDict, bc2: EdgeDict, r_edge: Edge, G_nx: nx.Graph, impact_treshold: float = 1e-5) -> dict[str, float]:
    """
    Takes two bc dictionnaries and measures the differences
    Warning they must have the same keys except from the r_edge

    Metrics:
        - max distance from impact, key: "dmax"
        - number of impacted edges, key: "nimpacts"
        - absolute differences sum, key: "sumdiffs"
        - neg differences sum,      key: "sumnegs"
        - pos differences sum,      key: "sumpos"
        - biggest pos change,       key: "maxdiff"
        - biggest neg change,       key: "mindiff"
        (res["sumdiffs"] = res["neg"] + res["sumpos"])
        - absolute dif sum divided by dist,     key: "sumbydist"
        - absolute dif sum divided by dist**-1, key: "sumbyinvdist"
    """
    res = {
        "dmax": 0,
        "nimpacts": 0,
        "sumdiffs": 0,
        "sumnegs": 0,
        "sumpos": 0,
        "maxdiff": 0,
        "mindiff": 0,
        "sumbydist": 0,
        "sumbyinvdist": 0
    }
    xs = nx.get_node_attributes(G_nx, "x")
    ys = nx.get_node_attributes(G_nx, "y")
    r_edge_coord = ((xs[r_edge[0]], ys[r_edge[0]]), (xs[r_edge[1]], ys[r_edge[1]]))
    for edge, v in bc1.items():
        if edge == r_edge:
            continue
        delta = bc2[edge] - v
        if abs(delta) > impact_treshold:
            edge_coord = ((xs[edge[0]], ys[edge[0]]), (xs[edge[1]], ys[edge[1]]))
            if delta > 0:
                res["sumpos"] += delta
                res["maxdiff"] = max(delta, res["maxdiff"])
            else:
                res["mindiff"] = min(delta, res["maxdiff"])
                delta = -delta
                res["sumnegs"] += delta
            res["sumdiffs"] += delta
            res["nimpacts"] += 1
            d = dist(r_edge_coord, edge_coord)
            res["dmax"] = max(d, res["dmax"])
            res["sumbydist"] += delta / (d+1e-5)
            res["sumbyinvdist"] += delta * d      
    print(res) 
    return res

def cpt_eBC_without_div(G_nx):
    """Returns for each edge, the number of shortest paths passing through it"""
    res = {}
    cpt, E = 0, len(G_nx.nodes)
    for s in G_nx.nodes:
        print(f"processing {cpt} out of {E}")
        for t in G_nx.nodes:
            if s == t:
                continue
            path_gen = nx.all_shortest_paths(G_nx, s, t, weight="weight")
            for path in path_gen:
                n1 = s
                for n2 in path[1:]:
                    res[(n1, n2)] = res[(n1, n2)] + 1 if (n1, n2) in res else 1
    return res

def efficiency(robust_list: RobustList, G_nx: nx.Graph):
    for attack in robust_list:
        n1, n2 = eval(attack[0])
        G_nx.remove_edge(n1, n2)
    return nx.global_efficiency(G_nx)