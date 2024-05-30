from Graph import Graph
from typ import RobustList, Edge, EdgeDict
from geo import dist

import json
import random as rd
import networkx as nx
import numpy as np
from copy import deepcopy

def is_cutable(G: Graph, nblocks: int, imb: float):
    if G._nx and G._nx.is_directed():
        ccs = G.get_sccs
    else:
        ccs = G.get_ccs
    if len(ccs) < nblocks:
        return True
    for cc1 in ccs[:nblocks]:
        for cc2 in ccs[:nblocks]:
            if len(cc1) + G._sizeV * imb < len(cc2) or len(cc1) - G._sizeV * imb > len(cc2):
                return True
    return False

def freq_attack(G: Graph, nblocks: int, ncuts: int, imb: float) -> Edge:      
    cut_union = []
    seen_seeds = []
    if is_cutable(G, nblocks, imb):
        for _ in range(ncuts):
            seed = rd.randint(0, 1044642763)
            while seed in seen_seeds:
                seed = rd.randint(0, 1044642763)
            seen_seeds.append(seed)
            G.kaffpa_cut(nblocks, imb, 0, seed, 2)
            cut_union += G.process_cut()
    else:
        print("not cutable")
        largest_cc = G.get_biggest_cc
        G_nx = G._nx.subgraph(largest_cc)
        # code de add node weights and relabel de preprocessing
        w_nodes = {}
        for node in list(G_nx.nodes):
            w_nodes[node] = 1
        nx.set_node_attributes(G_nx, w_nodes, "weight")
        sorted_nodes = sorted(G_nx.nodes())
        mapping = {old_node: new_node for new_node, old_node in enumerate(sorted_nodes)}
        G_nx = nx.relabel_nodes(G_nx, mapping)

        G_sub = Graph(nx=G_nx)
        res = freq_attack(G_sub, nblocks, ncuts, imb)
        n1 = list(mapping.keys())[list(mapping.values()).index(res[0])]
        n2 = list(mapping.keys())[list(mapping.values()).index(res[1])]
        if not n1 in G._nx.subgraph(largest_cc).nodes:
            print("not in subgraph")
        if not n2 in G._nx.subgraph(largest_cc).nodes:
            print("not in subgraph")
        if not (n1, n2) in G._nx.subgraph(largest_cc).edges:
            print(f"{n1, n2} not in edges (A)")
        if not (n2, n1) in G._nx.subgraph(largest_cc).edges:
            print(f"{n2, n1} not in edges (B)")
        return (n1, n2)
    
    if len(cut_union) == 0:
        print([len(scc) for scc in G.get_ccs])
    frequencies = {}
    for edge in cut_union:
        if edge in frequencies:
            frequencies[edge] += 1
        else:
            frequencies[edge] = 1
    return max(frequencies, key=frequencies.get)


def betweenness_attack(G: Graph, weighted: bool, subset: list[Edge] | None, approx: int | None, new: bool = False) -> Edge:
    bc = G.get_edge_bc(weighted=weighted, new=new, approx=approx)
    if subset:
        res, max_bc = None, 0
        for edge in subset:
            if edge in bc and bc[edge] > max_bc:
                max_bc = bc[edge]
                res = edge
        return res
    else:
        return max(bc, key=bc.get)


def random_attack(G: Graph, subset: list[Edge] | None) -> Edge:
    if subset:
        return rd.choice(subset)
    return rd.choice(list(G._nx.edges))


def maxdegree_attack(G: Graph, subset: list[Edge] | None) -> Edge:
    maxdegree, chosen_edge = 0, None
    if subset:
        for edge in subset:
            degree = G._nx.degree[edge[0]] * G._nx.degree[edge[1]]
            maxdegree = maxdegree if maxdegree > degree else degree
            chosen_edge = edge
    else:
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
    metric_scc: bool,
    ncuts: int = 1000,
    save: bool = True,
    subset: list[Edge] | None = None,
    weighted: bool = True,
    bc_approx: int | None = None,
    imb: float = 0.05,
    nblocks: int = 2
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

    def metric_procedure(metrics, chosen_edge):
        bc = G.get_edge_bc(weighted=weighted, new=True, approx=bc_approx) if metric_bc else None
        cc = len(G.get_biggest_cc) if metric_cc else None
        scc = len(G.get_biggest_scc) if metric_scc else None
        metrics.append((chosen_edge, bc, cc, scc))

    if order == "freq" and subset:
        raise ValueError(
            "Freq attack not available when considering only a subset of edges"
        )
    metrics, chosen_edge, chosen_edges, direct_save = [], None, [], False
    for i in range(k):
        print(f"processing the {i+1}-th attack over {k}, order: {order}")
        match order:
            case "bc":
                metric_procedure(metrics, chosen_edge)
                chosen_edge = betweenness_attack(G, weighted, subset, bc_approx, not metric_bc)
                G.remove_edge(chosen_edge[:2])
            case "freq":
                metric_procedure(metrics, chosen_edge)
                chosen_edge = freq_attack(G, nblocks, ncuts, imb)
                if not chosen_edge:
                    direct_save = True
                    break
                G.remove_edge(chosen_edge)
            case "deg":
                metric_procedure(metrics, chosen_edge)
                chosen_edge = maxdegree_attack(G, subset)
                G.remove_edge(chosen_edge)
            case "rd":
                metric_procedure(metrics, chosen_edges)
                G._nx = G.to_nx()
                chosen_edges = random_attack(G, subset)
    if not direct_save:
        metric_procedure(metrics, chosen_edge)
    if save:
        temp = metrics.copy()
        metrics = []
        for step in temp:
            edge = str(step[0]) if step[0] else None
            str_d = {str(k): v for k, v in step[1].items()} if step[1] else None
            cc = step[2] if metric_cc else None
            scc = step[3] if metric_scc else None
            metrics.append([edge, str_d, cc, scc])
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
    metric_scc: bool,
    ncuts: int,
    save: bool,
    subset: list[Edge] | None = None,
    weighted: bool = False,
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
        metric_scc,
        ncuts=ncuts,
        save=False,
        subset=subset,
        weighted=weighted,
    )
    # return the concat of the two metrics
    temp = tail.copy()
    tail = []
    for step in temp:
        str_d = {str(k): v for k, v in step[1].items()}
        tail.append([str(step[0]), str_d, step[2]])
    if save:
        with open(fp_save, "w") as saving_file:
            json.dump(metrics + tail, saving_file)
    else:
        return metrics + tail


def verify_integrity(fp: str, order: str, size: int) -> None:
    """ "
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
            if (
                isinstance(edge, tuple)
                and isinstance(edge[0], int)
                and isinstance(edge[1], int)
            ):
                removed_edges.append(edge)
            else:
                raise TypeError(
                    f"First element of the attack tuple must be the removed edge. Instead type {type(edge)} found"
                )
        else:
            assert edge == None
        if type(bc) != dict:
            raise TypeError(
                f"the betweenness centrality measures should be of type dict instead {type(bc)} is found"
            )
        for k, v in bc.items():
            if not isinstance(k, str) or not isinstance(v, float):
                raise TypeError(
                    f"the betweenness centrality measures should be of type dict[str, float] instead {type(bc)}[{type(k)}, {type(v)}] is found"
                )
        if type(cc) != int:
            raise TypeError(
                f"the biggest connected component measures should be of type int, instead {type(cc)} is found"
            )
    for j, edge in enumerate(removed_edges):
        if order == "bc":
            assert eval(max(data[j][1], key=data[j][1].get)) == edge
        assert not edge in data[j + 1][1]


def measure_scc_from_rlist(
    robust_list: RobustList, G_nx: nx.Graph
) -> list[list[int]]:
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
            try:
                G.remove_edge(edge[1], edge[0])
            except:
                pass
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

def measure_bc_impact_cumulative(
    r_edges: list[Edge],
    bcs: list[EdgeDict],
    G_nx: nx.Graph,
    save_path: str,
    impact_treshold: float = 1e-3,
) -> list[dict[str, float]]:
    """
    Takes output from preprocess_robust_import and measures the impacts
    in a cumulative way (aggregates the pertubated edges)
    except for sumdiffsnoC and dmax
    impact_treshold: float, determines the threshold for a eBC difference to be considered as a pertubation
    Warning they must have the same keys except from the r_edge

    Metrics:
        - max distance from impact, key: "dmax"
        - number of impacted edges, key: "nimpacts"
        - absolute differences sum, key: "sumdiffs"
        - absolute dif sum divided by dist,     key: "sumbydist"
    """
    res = []
    impact_dict = {
        "dmax": 0,
        "nimpacts": 0,
        "sumdiffs": 0,
        "sumbydist": 0,
        "sumdiffsnoC": 0
    }
    xs = nx.get_node_attributes(G_nx, "x")
    ys = nx.get_node_attributes(G_nx, "y")
    r_edges_coord = [((xs[r_edge[0]], ys[r_edge[0]]), (xs[r_edge[1]], ys[r_edge[1]])) for r_edge in r_edges[1:]]
    impacted_edges = []
    first_bc = bcs[0]
    for i, bc in enumerate(bcs[1:]):
        print(i)
        impact_dict["dmax"] = 0
        impact_dict["sumdiffsnoC"] = 0
        for edge, v in bc.items():
            if edge == r_edges[i]:
                continue
            delta = abs(first_bc[edge] - v)
            if delta > impact_treshold:
                if not edge in impacted_edges:    
                    impacted_edges.append(edge)
                    impact_dict["nimpacts"] += 1
                edge_coord = ((xs[edge[0]], ys[edge[0]]), (xs[edge[1]], ys[edge[1]]))
                impact_dict["sumdiffs"] += delta
                impact_dict["sumdiffsnoC"] += delta
                d = dist(r_edges_coord[i], edge_coord)/1000 # in kilometers
                impact_dict["dmax"] = max(d, impact_dict["dmax"])
                impact_dict["sumbydist"] += delta / (d + 1e-5)
            elif edge in impacted_edges:
                impacted_edges.remove(edge)
                impact_dict["nimpacts"] -= 1
        res.append(impact_dict.copy())
    with open(save_path, "w") as wfile:
        json.dump(res, wfile)

def efficiency(G_nx: nx.Graph):
    efficiency = {}
    G = G_nx.to_undirected() if G_nx.is_directed() else G_nx
    cpt, E = 0, len(G.edges)
    for edge in G.edges:
        efficiency[str(edge)] = nx.efficiency(G, edge[0], edge[1])
        cpt += 1
    return efficiency

def cascading_failure(G_nx: nx.Graph, redges: list[Edge], rtreshold: float, ltreshold: tuple[int, int] = (0, None), bc_dict: EdgeDict | None = None, approx: int | None= None) -> list[tuple[list[Edge], EdgeDict]]:
    """
    Simulates cascading failures, removes the edges in the list, computes the corresponding eBC
    and removes the edges with a eBC above the rthreshold.
    Repeats untill less than ltreshold edges are concerned.
    If bc_dict is set, than it's used as the first eBC
    """
    def cpt_fails(bc):
        l = []
        for k, v in bc.items():
            if v > rtreshold:
                l.append(k)
        return l
    def remove_edges(G, edgelist):
        for edge in edgelist:
            try:
                e = eval(edge)
            except:
                e = (eval(edge[0]), eval(edge[1]))
            try:
                G.remove_edge(e[0], e[1])
            except:
                G.remove_edge(e[1], e[0])
    remove_edges(G_nx, redges)
    last_bc = bc_dict if bc_dict else nx.edge_betweenness_centrality(G_nx, approx, weight="weight")
    res = [(redges, last_bc)]
    fails = cpt_fails(last_bc)
    rounds = ltreshold[1] if ltreshold[1] else 1000
    while len(fails) > ltreshold[0] or rounds > 0:
        rounds -= 1
        remove_edges(G_nx, fails)
        last_bc = nx.edge_betweenness_centrality(G_nx, approx, weight="weight")
        res.append((fails, last_bc))
    return res

def cpt_effective_resistance(G_nx: nx.Graph, weight: bool) -> float:
    if not nx.is_connected(G_nx):
        largest_cc = max(nx.connected_components(G_nx), key=len)
        print(f"the graph isn't connected, effective resistance will be computed on a component of size {len(largest_cc)} (real graph size is {len(G_nx.nodes)})")
        G_nx = G_nx.subgraph(largest_cc)
    return nx.effective_graph_resistance(G_nx, 'weight') if weight else nx.effective_graph_resistance(G_nx)
    