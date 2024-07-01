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
        self._id:         int         = id
        self._parent:     Node | None = parent
        self._is_leaf:    bool        = is_leaf
        self._children:   list[Node]  = []
        self._edge_union: Cut         = cut
        self._cuts:       list[Cut]   = [cut]
        self._diameter:     float        = 0

    def display_node(self):
        children_size = len(self._children) if self._children else 0
        cuts_size = len(self._cuts) if self._cuts else 0
        if self._is_leaf:
            return "Leaf " + str(self._id) + ", with " + str(cuts_size) + " cuts"
        return "Node " + str(self._id) + ", with " + str(children_size) + " children"

class CFTree:
    def __init__(self, cuts: Cuts, G_nx: nx.Graph, threshold: float):
        print('init tree...')
        self._last_id = 0
        self._root = Node(0, None, [], True)
        self._threshold = threshold
        self._cuts = cuts
        self._latitudes = nx.get_node_attributes(G_nx, "x")
        self._longitudes = nx.get_node_attributes(G_nx, "y")
        for cut in self._cuts:
            for edge in cut:
                if not edge[0] in self._latitudes or not edge[0] in self._longitudes:
                    raise ValueError(f"{edge[0]} has no geo coordinates")
                if not edge[1] in self._latitudes or not edge[1] in self._longitudes:
                    raise ValueError(f"{edge[0]} has no geo coordinates")
        self._avg_cut_size = sum([len(cut) for cut in self._cuts]) / len(self._cuts)
        print('...tree initialized')
    
    def __str__(self):
        def parcours_display(node):
            res = node.display_node()
            if not node._is_leaf:
                for child in node._children:
                    res += "\n   "
                    res += parcours_display(child)
            return res
        return parcours_display(self._root)

    @property
    def _get_new_id(self):
        self._last_id += 1
        return self._last_id

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
    
    def adapted_chamfer_distance(self, u1: Cut, u2: Cut) -> float:
        """Chamfer Distance adapted to modified BIRCH algorithm, so returns a lookalike chamfer distance to the  / union"""
        size = int(10 * self._avg_cut_size)
        if not u1 or len(u1) == 1:
            return 0
        if len(u1) > size:
            u1 = rd.sample(u1, size)
        if len(u2) > size:
            u2 = rd.sample(u2, size)
        return sum(self.chamfer_routine(u1, u2)) / size

    def insert(self, cut):
        # If the tree is empty, add the cut to the root
        if len(self._root._children) == 0:
            self._root._is_leaf = False
            self._root._children.append(Node(self._get_new_id, self._root, cut, True))
            return

        # Find the leaf node to insert the cut
        leaf_node, dist_to_node = self._find_leaf_node(self._root, cut, None)

        # Insert the cut into the leaf node
        self._insert_into_node(leaf_node, dist_to_node, cut)

    def _find_leaf_node(self, node: Node, cut, prev_dist):
        # If the node is a leaf, return it
        if node._is_leaf:
            if prev_dist:
                return node, prev_dist
            else:
                return node, self.adapted_chamfer_distance(node._edge_union, cut)
        # If the node is not a leaf, find the closest child
        min_distance = inf
        closest_child = None
        for child in node._children:
            distance = self.adapted_chamfer_distance(child._edge_union, cut)
            if distance < min_distance:
                min_distance = distance
                closest_child = child
 
        # Recursively find the leaf node
        return self._find_leaf_node(closest_child, cut, min_distance)

    def _insert_into_node(self, node: Node, distance, cut: Cut):
        # Update the node attributes
        node._cuts.append(cut)
        node._edge_union += cut
        remove_duplicates(node._edge_union)

        # If the node's diameter exceeds the threshold, split the node
        node._diameter = max(node._diameter, distance)
        if node._diameter > self._threshold:
            self._split_node(node)

    def _split_node(self, node: Node):
        # Create two new nodes
        new_node1 = Node(self._get_new_id, node._parent, [], True)
        new_node2 = Node(self._get_new_id, node._parent, [], True)
        new_node_list = [new_node1, new_node2]

        # Redistribute the data cuts between the two new nodes
        cuts_partition = self.kmeanspp(node._cuts, 2)
        for cut_id, part_id in enumerate(cuts_partition):
            self._insert_into_node(new_node_list[part_id], 0, node._cuts[cut_id])

        # Add the new nodes in the tree
        for new_node in new_node_list:
            node._parent._children.append(new_node)

    def kmeanspp(self, cuts: list[Cut], nrepr: int) -> list[int]:
        l = len(cuts)
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
                # a tester
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
    
    def merge_clusters(self, c1: Node, c2: Node, d: float):
        c2._id = c1._id
        c2._edge_union += c1._edge_union
        c1._edge_union += c2._edge_union
        remove_duplicates(c1._edge_union)
        remove_duplicates(c2._edge_union)
        c1._cuts += c2._cuts
        c2._cuts += c1._cuts
        remove_duplicates(c1._cuts)
        remove_duplicates(c2._cuts)
        c1._diameter, c2._diameter = d, d

    def activate_clustering(self):
        for i, cut in enumerate(self._cuts):
            self.insert(cut)
            print(f"cut {i+1} out of {len(self._cuts)} processed")
        # print("clustering done, merging close clusters...")
        # for c1 in self._root._children:
        #     for c2 in self._root._children:
        #         if c1._id != c2._id:
        #             dist = self.adapted_chamfer_distance(c1._edge_union, c2._edge_union)
        #             if dist < self._threshold / 10:
        #                 self.merge_clusters(c1, c2, dist)

    def retrieve_cluster(self) -> list[list[Cut]]:
        clusters = []
        for cluster in self._root._children:
            clusters.append(cluster._cuts)
        return clusters