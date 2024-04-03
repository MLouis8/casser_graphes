from typ import Cuts, Cut, Classes
from Graph import Graph
from typing import Any, Optional

from utils import gen_to_list

import json
import networkx as nx
import random as rd
import numpy as np
from math import inf, sqrt

class CutsClassification:
    def __init__(self, cuts: Cuts, G_nx: nx.Graph) -> None:
        self._cuts = cuts
        self._levels = None
        self._nodes = G_nx.nodes(data=True)

    def save_last_classes(self, filepath: str):
        with open(filepath, "w") as write_path:
            json.dump(self._levels, write_path)

    def distance(self, c1: Cut, c2: Cut, tdistance: str) -> float:
        """
        Distance between two cuts based on geographical proximity.
        First for each edge of c1 we find the closest (Euclidially) corresponding edge in c2.
        Then distance is defined as:
            - max: the largest value found
            - sum: sum of the values
            - sum squarred: sum of the squarred values
            - inter: only distance of 0 is kept
        """
        match tdistance:
            case "max": d_cut = lambda l: max(l)
            case "sum": d_cut = lambda l: sum(l)
            case "squared sum": d_cut = lambda l: sum([e**2 for e in l])
            case "inter":
                inter = 0
                for ele in c2:
                    if ele in c1:
                        inter += 1
                return int((inter / (len(c1) + len(c2)))*100)
            case "var": d_cut = lambda l: int(np.var(l))
            case _: raise ValueError("wrong distance parameter")
        l, seen = [], []
        for edge1 in c1:
            best_distance = inf
            x1 = self._nodes[edge1[0]]["x"] + self._nodes[edge1[1]]["x"] / 2
            y1 = self._nodes[edge1[0]]["y"] + self._nodes[edge1[1]]["y"] / 2
            for edge2 in c2:
                if not edge2 in seen:
                    x2 = self._nodes[edge2[0]]["x"] + self._nodes[edge2[1]]["x"] / 2
                    y2 = self._nodes[edge2[0]]["y"] + self._nodes[edge2[1]]["y"] / 2
                    d_edge = sqrt((x1-x2)**2 + (y1-y2)**2)
                    if d_edge < best_distance:
                        best_distance = d_edge
            if best_distance == inf:
                l.append(0)
            elif best_distance == 0:
                l.append(-1)
            else:
                l.append(int(best_distance**(-1)))
        return d_cut([1000 if e == -1 else e for e in l])

    def cluster_louvain(self, distance_type: str, treshold: bool=False) -> list[Any]:
        G = nx.Graph()
        weights = []
        for k, v in self._cuts.items():
            for kprime, vprime in self._cuts.items():
                if not (k,  kprime) in G.edges and not (kprime, k) in G.edges and k != kprime:
                    w = self.distance(v, vprime, distance_type)
                    weights.append(w)
                    if not treshold or w >= 10000:
                        G.add_edge(k, kprime, weight=w)
        print(min(weights), max(weights), np.mean(weights), np.var(weights))
        self._levels = gen_to_list(nx.community.louvain_partitions(G))

    def get_class_level(self):
        if not self._levels:
            raise ValueError("classification must be done first, try calling cluster_louvain")
        elif len(self._levels) == 1:
            return self._levels[0]
        return self._levels[-2]
    
    def get_levels(self):
        if not self._levels:
            raise ValueError("classification must be done first, try calling cluster_louvain")
        return self._levels

class HomemadeClassification:
    def __init__(self) -> None:
        pass 

    def intersection_criterion(c1: Cut, c2: Cut, eps: float) -> bool:
        """Takes as parameters Cut objects and return whether their intersection is big enough according to epsilon"""
        obj, cpt = len(c2) * eps, 0
        for ele in c2:
            if ele in c1:
                cpt += 1
            if cpt > obj:
                return True
        return cpt > obj


    def neighbor_criterion(c1: Cut, c2: Cut, G_kp: Graph, k: int) -> bool:
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
        c1: Cut, c2: Cut, G_kp: Graph, p: float, k: int
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
        c1: Cut, c2: Cut, G_kp: Graph, G_nx: Any, t: float
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
    def proximity(c1: Cut, c2: Cut) -> float:
        """proximity criterion based on intersection"""
        inter = 0
        for ele in c2:
            if ele in c1:
                inter += 1
        return inter / (len(c1) + len(c2))
    
    def representant_method(
        self,
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
                criterion = lambda u, v: self.intersection_criterion(u, v, p)
            case "neighbor":
                if not G_kp:
                    raise TypeError(
                        "A Graph should be passed as argument for the neighbor criterion"
                    )
                criterion = lambda u, v: self.neighbor_criterion(u, v, G_kp, n)
            case "mixed":
                if not G_kp:
                    raise TypeError(
                        "A Graph should be passed as argument for the mixed criterion"
                    )
                criterion = lambda u, v: self.mixed_criterion(u, v, G_kp, p, n)
            case "geographical":
                if not G_kp:
                    raise TypeError(
                        "A Graph should be passed as argument for the mixed criterion"
                    )
                criterion = lambda u, v: self.geographical_criterion(u, v, G_kp, G_nx, t)
        classes: Classes = []
        for k, cut in self._cuts.items():
            classified = False
            for cls in classes:
                if criterion(self._cuts[cls[0]], cut):
                    cls.append(k)
                    classified = True
                    break
            if not classified:
                classes.append([k])
        return classes
    
    def iterative_division(self, n: int, treshold: float) -> Classes:
        """
        classification method with different approach than representant
        
        Here the cuts start as classed in the same category and are then sorted according to their proximity to each other
        """
        potential_rpz, rpz, to_classify = list(self._cuts.keys()), [], list(self._cuts.keys())
        for _ in range(n):
            rpz.append(potential_rpz.pop(rd.randint(0, len(potential_rpz) - 1)))
            to_remove = []
            for p_rpz in potential_rpz:
                if self.proximity(self._cuts[rpz[-1]], self._cuts[p_rpz]) > treshold:
                    to_remove.append(p_rpz)
            for elem in to_remove:
                potential_rpz.remove(elem)
        classes = [[rp] for rp in rpz]
        for elem in to_classify:
            classes[
                max(range(len(rpz)), key=(lambda k: self.proximity(self._cuts[rpz[k]], self._cuts[elem])))
            ].append(elem)
        return classes