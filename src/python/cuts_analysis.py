from scipy.stats import pearsonr
import networkx as nx
from Graph import Graph
import random as rd
from typing import Optional, Any
from numpy import mean
from math import inf
from typ import KCuts, EdgeDict, EdgeDictStr, Cuts, Classes

def determine_edge_frequency(G: Graph, C: dict[str, KCuts]) -> EdgeDict:
    """
    Function for determining edge frequency in cuts

    Paramters:
        G: Graph
        C: dictionary of cuts indexed by cut name: {'cut_name': (edgecut: int, blocks: list[int])}
    """
    frequencies: EdgeDict = {}

    for _, val in C.items():
        G.set_last_results(*val)
        p_cut = G.process_cut(weight=True)
        for edge in p_cut:
            if edge[:2] in frequencies:
                frequencies[edge[:2]] += 1
            else:
                frequencies[edge[:2]] = 1

    # removing doublons
    freq = frequencies.copy()
    for n1, n2 in freq:
        if (n2, n1) in frequencies.keys() and (n1, n2) in frequencies.keys():
            frequencies.pop((n2, n1))
    return frequencies


def cpt_cuts_correlation(edge_count: EdgeDict, measure: EdgeDictStr):
    """
    Analyses the correlation between the edges cut frequency and an arbitrary measure.

    Parameters:
        edge_count: dict(edge (tuple[int, int]): measure (float|int))
        measure: dict(edge str(tuple[int, int]): measure (float|int))

    Warning dicts keys must correspond.
    """
    x, y, nb_count = [], [], len(edge_count.keys())
    for edge in edge_count.keys():
        x.append(edge_count[edge] / nb_count)
        edge_converted = str((edge[0], edge[1], 0))
        edge_converted_r = str((edge[1], edge[0], 0))
        try:
            y.append(measure[edge_converted])
        except:
            y.append(measure[edge_converted_r])
    return pearsonr(x, y)


def get_n_biggest_freq(freq: EdgeDict, n: int) -> EdgeDict:
    """Return dict of edges: count of the n most frequent edges"""
    chosen = {}
    for k, v in freq.items():
        chosen[k] = v
        if len(chosen.keys()) > n:
            chosen.pop(min(chosen, key=chosen.get))
    return chosen


def to_Cut(xadj: list[int], adjncy: list[int], blocks: list[int]):
    edges = []
    for i in range(1, len(xadj)):
        for j in range(xadj[i - 1], xadj[i]):
            edges.append((i - 1, adjncy[j]))
    cut_edges = []
    for edge in edges:
        if blocks[edge[0]] != blocks[edge[1]]:
            if not edge in cut_edges and not (edge[1], edge[0]) in cut_edges:
                cut_edges.append(edge)
    return cut_edges


def intersection_criterion(c1: list[tuple[int, int]], c2: list[tuple[int, int]], eps: float) -> bool:
    """Takes as parameters Cut objects and return whether their intersection is big enough according to epsilon"""
    obj, cpt = len(c2) * eps, 0
    for ele in c2:
        if ele in c1:
            cpt += 1
        if cpt > obj:
            return True
    return cpt > obj


def neighbor_criterion(c1: list[tuple[int, int]], c2: list[tuple[int, int]], G_kp: Graph, k: int) -> bool:
    """
    Takes as paramerters Cut objects and return wheter their local closeness is big enough

    In the representant method, c1 is the representant
    """
    for edge2 in c2:
        flag = False
        for edge1 in c1:
            if G_kp.closer_than_k_edges(edge1, edge2, k):
                flag = True
                break
        if not flag:
            return False
    return True


def mixed_criterion(
    c1: list[tuple[int, int]], c2: list[tuple[int, int]], G_kp: Graph, p: float, k: int
) -> bool:
    obj, cpt = len(c2) * p, 0
    for edge2 in c2:
        for edge1 in c1:
            if G_kp.closer_than_k_edges(edge1, edge2, k):
                cpt += 1
                break
        if cpt > obj:
            return True
    return cpt > obj

def geographical_criterion(
    c1: list[tuple[int, int]], c2: list[tuple[int, int]], G_kp: Graph, G_nx: Any, t: float
) -> bool:
    """
    Looks at the (lon, lat) values to determine wheter the edges are close enough
    Since only 2/3 of the nodes are located, it skips the non labelled nodes

    k is the geographical treshold
    """
    for edge2 in c2:
        flag = False
        try:
            lon1 = G_nx.nodes(data=True)[edge2[0]]["lon"] + G_nx.nodes(data=True)[edge2[1]]["lon"] / 2
            lat1 = G_nx.nodes(data=True)[edge2[0]]["lat"] + G_nx.nodes(data=True)[edge2[1]]["lat"] / 2
            for edge1 in c1:
                try:
                    lon2 = G_nx.nodes(data=True)[edge1[0]]["lon"] + G_nx.nodes(data=True)[edge1[1]]["lon"] / 2
                    lat2 = G_nx.nodes(data=True)[edge1[0]]["lat"] + G_nx.nodes(data=True)[edge1[1]]["lat"] / 2
                    if abs(lon1-lon2) < t and abs(lat1-lat2) < t:
                        flag = True
                except:
                    continue
        except:
            continue
        if not flag:
            return False
    return True            

def proximity(c1: list[tuple[int, int]], c2: list[tuple[int, int]]) -> float:
    """proximity criterion based on intersection"""
    inter = 0
    for ele in c2:
        if ele in c1:
            inter += 1
    return inter / (len(c1) + len(c2))

def distance(c1: list[tuple[int, int]], c2: list[tuple[int, int]], G_nx: nx.Graph, distance: str="max") -> float:
    """
    Distance between two cuts based on geographical proximity.
    Since it's not always defined, skips not defined edges.
    First for each edge of c1 we find the closest corresponding edge in c2.
    Then distance is defined as:
        - max: the largest value found
        - mean: the mean of the values
    Too much skipped edges, geometrical distance should be preferred
    """
    match distance:
        case "max": d_cut = lambda l: max(l)
        case "mean": d_cut = lambda l: mean(l)
        case _: raise ValueError("distance parameter should be either max or mean")
    l = []
    for edge1 in c1:
        best_distance = inf
        x1, y1 = G_nx.nodes(data=True)[edge1[0]]["x"], G_nx.nodes(data=True)[edge1[0]]["y"]
        for edge2 in c2:
            x2, y2 = G_nx.nodes(data=True)[edge1[0]]["x"], G_nx.nodes(data=True)[edge2[0]]["y"]
            d_edge = max(abs(x1-x2), abs(y1-y2))
            best_distance = d_edge if d_edge < best_distance else best_distance
        l.append(best_distance)
        first = False
    return d_cut(l)

def distance_geo(c1: list[tuple[int, int]], c2: list[tuple[int, int]], G_nx: nx.Graph, distance: str="max") -> float:
    """
    Distance between two cuts based on geographical proximity.
    Since it's not always defined, skips not defined edges.
    First for each edge of c1 we find the closest corresponding edge in c2.
    Then distance is defined as:
        - max: the largest value found
        - mean: the mean of the values
    Too much skipped edges, geometrical distance should be preferred
    """
    match distance:
        case "max": d_cut = lambda l: max(l)
        case "mean": d_cut = lambda l: mean(l)
        case _: raise ValueError("distance parameter should be either max or mean")
    l = []
    first = True
    skipped1, skipped2 = 0, 0
    for edge1 in c1:
        best_distance = inf
        try:
            lon1, lat1 = G_nx.nodes(data=True)[edge1[0]]["lon"], G_nx.nodes(data=True)[edge1[0]]["lat"]
        except:
            skipped1 += 1
            continue
        for edge2 in c2:
            try:
                lon2, lat2 = G_nx.nodes(data=True)[edge1[0]]["lon"], G_nx.nodes(data=True)[edge2[0]]["lat"]
            except:
                if first:
                    skipped2 += 1
                continue
            
            d_edge = max(abs(lon1-lon2), abs(lat1-lat2))
            best_distance = d_edge if d_edge < best_distance else best_distance
        l.append(best_distance)
        first = False
    return d_cut(l), l, skipped1, skipped2

def representant_method(
    cuts: Cuts,
    p: float = 0.5,
    n: int = 3,
    t: float = 0.05,
    criterion_name: str = "intersection",
    G_kp: Optional[Graph] = None,
    G_nx: Optional[Any] = None
) -> Classes:
    """
    Takes as parameter a list of Cut objects, returns a list of list of Cut objects
    corresponding to the cuts after classification according to the representant method and
    the criterion.

    Available criteria:
        intersection (default): classify according to the number of same moment
        neighbor: classify according to the edge closeness
        mixed: a mix of the two above
        geographical: using longitute and latitude
    """
    match criterion_name:
        case "intersection":
            criterion = lambda u, v: intersection_criterion(u, v, p)
        case "neighbor":
            if not G_kp:
                raise TypeError(
                    "A Graph should be passed as argument for the neighbor criterion"
                )
            criterion = lambda u, v: neighbor_criterion(u, v, G_kp, n)
        case "mixed":
            if not G_kp:
                raise TypeError(
                    "A Graph should be passed as argument for the mixed criterion"
                )
            criterion = lambda u, v: mixed_criterion(u, v, G_kp, p, n)
        case "geographical":
            if not G_kp:
                raise TypeError(
                    "A Graph should be passed as argument for the mixed criterion"
                )
            criterion = lambda u, v: geographical_criterion(u, v, G_kp, G_nx, t)
    classes: Classes = []
    for k, cut in cuts.items():
        classified = False
        for cls in classes:
            if criterion(cuts[cls[0]], cut):
                cls.append(k)
                classified = True
                break
        if not classified:
            classes.append([k])
    return classes


def best_representant(cuts, p=None, criterion_name="intersection", G_kp=None):
    """
    Takes as parameter a list of Cut objects, returns a list of list of Cut objects
    corresponding to the cuts after classification according to the best representant method and
    the criterion.

    Instead of just selecting a random representant we change representant untill the best is taken.
    VERY BAD
    Available criteria:
        intersection (default): classify according to the number of same moment
        neighbor: classify according to the edge closeness
    """
    match criterion_name:
        case "intersection":
            criterion = lambda u, v: intersection_criterion(u, v, p)
        case "neighbor":
            criterion = lambda u, v: neighbor_criterion(u, v, G_kp, p)
    classes = []
    for k, cut in cuts.items():
        classified = False
        # we check if we can classify it in the existing classes
        for cls in classes:
            covers_rpz, not_covered = True, []
            for rpz in cls[0]:
                if criterion(cuts[rpz], cut):
                    # it's related to a representant so it becomes a member of the class
                    if not classified:
                        cls[1].append(k)
                        classified = True
                else:
                    covers_rpz = False
                    not_covered.append(rpz)
            if classified:
                if not covers_rpz:
                    # we remove representatives that don't fit anymore
                    for rpz in not_covered:
                        cls[0].remove(rpz)
                else:
                    # we compare the new element with the rest of the member to check whether it's a representative
                    is_rpz = True
                    for member in cls[1]:
                        if not member in cls[0] and not criterion(cuts[member], cut):
                            is_rpz = False
                    if is_rpz:
                        cls[0].append(k)
        # no existing classes fit, so it becomes a represatative of a new class
        if not classified:
            classes.append(([k], [k]))
    return classes


def iterative_division(cuts: Cuts, n: int, treshold: float) -> Classes:
    potential_rpz, rpz, to_classify = list(cuts.keys()), [], list(cuts.keys())
    for _ in range(n):
        rpz.append(potential_rpz.pop(rd.randint(0, len(potential_rpz) - 1)))
        to_remove = []
        for p_rpz in potential_rpz:
            if proximity(cuts[rpz[-1]], cuts[p_rpz]) > treshold:
                to_remove.append(p_rpz)
        for elem in to_remove:
            potential_rpz.remove(elem)
    classes = [[rp] for rp in rpz]
    for elem in to_classify:
        classes[
            max(range(len(rpz)), key=(lambda k: proximity(cuts[rpz[k]], cuts[elem])))
        ].append(elem)
    return classes


def get_connected_components(G_kp: Graph) -> Any:
    """Get connected components from KaHIP graph using NetworkX"""
    G_nx = G_kp.to_nx()
    cut = G_kp.process_cut()
    G_nx.remove_edges_from(cut)
    return nx.connected_components(G_nx)

def classify_by_connected_components(cc: dict[str, list[int]], liberty: int=3) -> Classes:
    classes: Classes = []
    for cut_name, co_cpnts in cc.items():
        classified = False
        for class_id, cls in enumerate(classes):
            if cc[cls[0]][:liberty] == co_cpnts[:liberty]:
                classified = True
                break
        if classified:
            classes[class_id].append(cut_name)
        else:
            classes.append([cut_name])
    return classes

def cluster_louvain(cuts: Cuts, G_nx: nx.Graph) -> list[set]:
    G = nx.Graph()
    for k, v in cuts.items():
        for kprime, vprime in cuts.items():
            G.add_edge(k, kprime, weight=distance(v, vprime, G_nx, distance="mean"))
    return [list(c) for c in nx.community.louvain_communities(G)]
