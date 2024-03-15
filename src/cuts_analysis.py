from scipy.stats import pearsonr

def determine_edge_frequency(G, C):
    """
    Function for determining edge frequency in cuts
    
    Paramters:
        G: Graph
        C: dictionary of cuts indexed by cut name: {'cut_name': (edgecut: int, blocks: list[int])}
    """
    frequencies = {}
    
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
    for (n1, n2) in freq:
        if (n2, n1) in frequencies.keys() and (n1, n2) in frequencies.keys():
            frequencies.pop((n2, n1))
    return frequencies

def cpt_cuts_correlation(edge_count, measure):
    """
    Analyses the correlation between the edges cut frequency and an arbitrary measure.

    Parameters:
        edge_count: dict(edge (tuple[int, int]): count (int)) (frequency)
        measure: dict(edge (tuple[int, int]): measure (float|int))

    Warning dicts keys must correspond.
    """
    x, y, nb_count = [], [], len(edge_count.keys())
    for edge in edge_count.keys():
        x.append(edge_count[edge]/nb_count)
        edge_converted = str((edge[0], edge[1], 0))
        edge_converted_r = str((edge[1], edge[0], 0))
        try:
            y.append(measure[edge_converted])
        except:
            y.append(measure[edge_converted_r])
    return pearsonr(x, y)

def get_n_biggest_freq(freq, n):
    """Return dict of edges: count of the n most frequent edges"""
    chosen = {}
    for k, v in freq.items():
        chosen[k] = v
        if len(chosen.keys()) > n:
            chosen.pop(min(chosen, key=chosen.get))
    return chosen

def to_Cut(xadj, adjncy, blocks):
    edges = []
    for i in range(1, len(xadj)):
        for j in range(xadj[i - 1], xadj[i]):
            edges.append((i - 1, adjncy[j]))
    cut_edges = set()
    for edge in edges:
        if blocks[edge[0]] != blocks[edge[1]]:
            if not edge in cut_edges:
                cut_edges.add(edge)
        return cut_edges

def intersection_criterion(c1: set[int], c2: set[int], eps=0.5):
    """Takes as parameters Cut objects and return whether their intersection is big enough according to epsilon"""
    return len(c1.intersection(c2)) >= eps * len(c1)

def neighbor_criterion(c: set[int], c2: set[int], k=8):
    """Takes as paramerters Cut objects and return wheter their local closeness is big enough"""
    for edge2 in c2:
        for edge in c:
            if neighbor_of_rank_k(edge, edge2, k):
                continue
            else:
                return False
    return True

def representant_method(cuts, p=None, criterion="intersection"):
    """
    Takes as parameter a list of Cut objects, returns a list of list of Cut objects
    corresponding to the cuts after classification according to the representant method and 
    the criterion.

    Available criteria:
        intersection (default): classify according to the number of same moment
        neighbor: classify according to the edge closeness
    """
    match criterion:
        case "intersection": 
            criterion = intersection_criterion
        case "neighbor": 
            criterion = neighbor_criterion 
    for cut in cuts:


