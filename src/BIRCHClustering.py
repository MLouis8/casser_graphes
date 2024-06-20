import numpy as np
import networkx as nx
from geo import dist
from math import inf
from typ import Cut, Cuts
import random as rd

def remove_duplicates(edges: Cut):
    to_remove = []
    for i, e1 in enumerate(edges):
        for e2 in edges[i+1:]:
            if e1 == e2:
                to_remove.append(e2)
                break
    for edge in to_remove:
        edges.remove(edge)

class Node:
    def __init__(self, id, parent, cut, is_leaf):
        self.id:         int         = id
        self.parent:     Node | None = parent
        self.is_leaf:    bool        = is_leaf
        self.children:   list[Node]  = []
        self.edge_union: Cut         = cut
        self.cuts:       list[Cut]   = [cut]
        self.radius:     float        = 0

    def display_node(self):
        children_size = len(self.children) if self.children else 0
        cuts_size = len(self.cuts) if self.cuts else 0
        if self.is_leaf:
            return "Leaf " + str(self.id) + ", with " + str(cuts_size) + " cuts"
        return "Node " + str(self.id) + ", with " + str(children_size) + " children"

class CFTree:
    def __init__(self, cuts: Cuts, G_nx: nx.Graph, threshold=5000):
        print('init tree...')
        self.last_id = 0
        self.root = Node(0, None, [], True)
        self.threshold = threshold
        self._cuts = cuts
        self._latitudes = nx.get_node_attributes(G_nx, "x")
        self._longitudes = nx.get_node_attributes(G_nx, "y")
        for cut in self._cuts:
            for edge in cut:
                if not edge[0] in self._latitudes or not edge[0] in self._longitudes:
                    raise ValueError(f"{edge[0]} has no geo coordinates")
                if not edge[1] in self._latitudes or not edge[1] in self._longitudes:
                    raise ValueError(f"{edge[0]} has no geo coordinates")
        print('...tree initialized')
    
    def __str__(self):
        def parcours_display(node):
            res = node.display_node()
            if not node.is_leaf:
                for child in node.children:
                    res += "\n   "
                    res += parcours_display(child)
            return res
        return parcours_display(self.root)

    @property
    def get_new_id(self):
        self.last_id += 1
        return self.last_id

    def chamfer_routine(self, c1: Cut, c2: Cut) -> float:
        l = []
        for e1 in c1:
            best_distance = inf
            e1n1 = (self._latitudes[e1[0]], self._longitudes[e1[0]])
            e1n2 = (self._latitudes[e1[1]], self._longitudes[e1[1]])
            for e2 in c2:
                e2n1 = (self._latitudes[e2[0]], self._longitudes[e2[0]])
                e2n2 = (self._latitudes[e2[1]], self._longitudes[e2[1]])
                d_edge = dist((e1n1, e1n2), (e2n1, e2n2))
                if d_edge < best_distance:
                    best_distance = d_edge
            l.append(best_distance)
        return l
    
    def chamfer_distance(self, c1: Cut, c2: Cut) -> float:
        """Chamfer Distance between two cuts based on geographical distance."""
        return sum(self.chamfer_routine(c1, c2)) + sum(self.chamfer_routine(c2, c1))
    
    def adapted_chamfer_distance(self, union: Cut, c: Cut) -> float:
        """Chamfer Distance adapted to modified BIRCH algorithm, so returns a lookalike chamfer distance to the  / union"""
        if not union:
            return 0
        if len(union) > 10*len(c):
            print('union too big')
            union = rd.sample(union, 10*len(c))
        return sum(self.chamfer_routine(union, c)) / len(union)
        # return max(self.chamfer_routine(union, c)) # a tester

    def insert(self, cut):
        # If the tree is empty, add the cut to the root
        if len(self.root.children) == 0:
            self.root.is_leaf = False
            self.root.children.append(Node(self.get_new_id, self.root, cut, True))
            return

        # Find the leaf node to insert the cut
        leaf_node, dist_to_node = self._find_leaf_node(self.root, cut, None)

        # Insert the cut into the leaf node
        self._insert_into_node(leaf_node, dist_to_node, cut)

    def _find_leaf_node(self, node: Node, cut, prev_dist):
        # If the node is a leaf, return it
        if node.is_leaf:
            if prev_dist:
                return node, prev_dist
            else:
                return node, self.adapted_chamfer_distance(node.edge_union, cut)

        # If the node is not a leaf, find the closest child
        min_distance = inf
        closest_child = None
        for child in node.children:
            distance = self.adapted_chamfer_distance(child.edge_union, cut)
            if distance < min_distance:
                min_distance = distance
                closest_child = child
 
        # Recursively find the leaf node
        return self._find_leaf_node(closest_child, cut, min_distance)

    def _insert_into_node(self, node: Node, distance, cut: Cut):
        # Update the node attributes
        node.cuts.append(cut)
        node.edge_union += cut
        remove_duplicates(node.edge_union)

        # If the node's radius exceeds the threshold, split the node
        node.radius = max(node.radius, distance)
        if node.radius > self.threshold:
            self._split_node(node)

    def _split_node(self, node: Node):
        # Create two new nodes
        new_node1 = Node(self.get_new_id, node.parent, [], True)
        new_node2 = Node(self.get_new_id, node.parent, [], True)
        new_node_list = [new_node1, new_node2]

        # Redistribute the data cuts between the two new nodes
        cuts_partition = self.kmeanspp(node.cuts, 2)
        for cut_id, part_id in enumerate(cuts_partition):
            self._insert_into_node(new_node_list[part_id], 0, node.cuts[cut_id])

        # Add the new nodes in the tree
        for new_node in new_node_list:
            node.parent.children.append(new_node)

    def kmeanspp(self, cuts: list[Cut], nrepr: int) -> list[int]:
        repr = [cuts.pop(rd.randrange(len(cuts)))]
        d = lambda cut, rs: min([self.chamfer_distance(cut, r) for r in rs])
        available_cuts = cuts.copy()
        for _ in range(nrepr-1):
            distances = {str(c): d(c, repr) for c in available_cuts}
            sum_distances = sum(distances.values())
            new_repr = rd.choices(available_cuts, [distances[str(c)] / sum_distances for c in available_cuts])[0]
            repr.append(new_repr)
            available_cuts.remove(new_repr)

        partition = []
        for cut in cuts:
            try:
                partition.append(repr.index(cut))
            except:
                best_dist = inf
                best_repr = None
                for repr_id, r in enumerate(repr):
                    cdist = self.chamfer_distance(cut, r)
                    if cdist < best_dist:
                        best_dist = cdist
                        best_repr = repr_id
                partition.append(best_repr)
        return partition

    def activate_clustering(self):
        for i, cut in enumerate(self._cuts):
            self.insert(cut)
            print(f"cut {i+1} out of {len(self._cuts)} processed")

    def retrieve_cluster(self) -> list[list[Cut]]:
        res = []
        for cluster in self.root.children:
            res.append(cluster.cuts)
        return res