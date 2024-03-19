from scipy.stats import pearsonr
import networkx as nx
from Graph import Graph
import random as rd
from typing import Optional

## Type Aliases ##
KCuts = tuple[int, list[int]]  # cuts under KaHIP format
EdgeDict = dict[tuple[int, int], int]  # common edge dict
EdgeDictStr = dict[str, int]  # edge dict after json import
Cuts = dict[
    str, list[tuple[int, int]]
]  # cuts after post processing, the name of the cut maps to the list of edges cut
Classes = list[list[str]]  # list of list of names of cuts


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
            if not edge in cut_edges:
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


def proximity(c1: list[tuple[int, int]], c2: list[tuple[int, int]]) -> float:
    """proximity criterion based on intersection"""
    inter = 0
    for ele in c2:
        if ele in c1:
            inter += 1
    return inter / (len(c1) + len(c2))


def representant_method(
    cuts: Cuts,
    p: float = 0.5,
    n: int = 3,
    criterion_name: str = "intersection",
    G_kp: Optional[Graph] = None,
) -> Classes:
    """
    Takes as parameter a list of Cut objects, returns a list of list of Cut objects
    corresponding to the cuts after classification according to the representant method and
    the criterion.

    Available criteria:
        intersection (default): classify according to the number of same moment
        neighbor: classify according to the edge closeness
        mixed: a mix of the two above
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


def measure_balance_artefacts(G_kp: Graph, treshold: int = 5) -> list[int]:
    cmpnts = G_kp.cpt_connex_components()
    artefacts = []
    for cmpnt in cmpnts:
        if len(cmpnt) < treshold:
            artefacts.append(cmpnt)
    print(f"there are {len(artefacts)} balancing artefacts smaller than {treshold}")
    return artefacts
