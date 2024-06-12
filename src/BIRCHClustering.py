import numpy as np
import networkx as nx
from geo import dist
from math import inf
from typ import Cut, Cuts

class Node:
    def __init__(self, parent=None, is_leaf=False, cut=None):
        self.parent = parent
        self.is_leaf = is_leaf
        self.children = []
        self.edge_union: Cut | None = cut
        self.cuts: list[Cut] = cut
        self.radius = 0

class CFTree:
    def __init__(self, cuts: Cuts, G_nx: nx.Graph, branching_factor=50, threshold=0.7, is_leaf=True):
        self.root = Node(is_leaf=is_leaf)
        self.branching_factor = branching_factor
        self.threshold = threshold

        self._cuts = cuts
        # self._levels = None
        # self._nodes = G_nx.nodes(data=True)
        self._latitudes = nx.get_node_attributes(G_nx, "x")
        self._longitudes = nx.get_node_attributes(G_nx, "y")
    
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
        return l
    
    def chamfer_distance(self, c1: Cut, c2: Cut) -> float:
        """Chamfer Distance between two cuts based on geographical distance."""
        return sum(self.chamfer_routine(c1, c2)) + sum(self.chamfer_routine(c2, c1))
    
    def adapted_chamfer_distance(self, union: Cut, c: Cut) -> float:
        """Chamfer Distance adapted to modified BIRCH algorithm, so returns a lookalike chamfer distance to the  / union"""
        return sum(self.chamfer_routine(union, c)) / len(union)
    # return max(self.chamfer_routine(union, c))

    def insert(self, cut):
        # If the tree is empty, create a new leaf node
        if len(self.root.children) == 0:
            leaf_node = Node(parent=self.root, is_leaf=True, n=1, cut=cut)
            self.root.children.append(leaf_node)
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
            distance = self.chamfer_distance(child.edge_union, cut)
            if distance < min_distance:
                min_distance = distance
                closest_child = child
 
        # Recursively find the leaf node
        return self._find_leaf_node(closest_child, cut, min_distance)

    def _insert_into_node(self, node: Node, distance, cut: Cut):
        # Update the node attributes
        node.n += 1
        node.edge_union = list(set(node.edge_union + cut))

        # If the node's radius exceeds the threshold, split the node
        node.radius = max(node.radius, distance)
        if node.radius > self.threshold:
            self._split_node(node)

    def _split_node(self, node):
        # Create two new nodes
        new_node1 = Node(parent=node.parent, is_leaf=node.is_leaf)
        new_node2 = Node(parent=node.parent, is_leaf=node.is_leaf)

        # Redistribute the data cuts between the two new nodes

        cuts_partition = []
        for cut in cuts_partition:
            self._insert_into_node(new_node1, cut)
        for cut in cuts:
            if cut not in cuts_partition:
                self._insert_into_node(new_node2, cut)

        # Replace the original node with the two new nodes in the parent node
        index = node.parent.children.index(node)
        node.parent.children[index] = new_node1
        node.parent.children.insert(index+1, new_node2)

        # If the parent node now has too many children, split the parent node
        if len(node.parent.children) > self.branching_factor:
            self._split_node(node.parent)

    def global_clustering(self, n_clusters):
        # Get the leaf nodes
        leaf_nodes = self._get_leaf_nodes(self.root)

        # Initialize each leaf node as a separate cluster
        clusters = [[leaf_node] for leaf_node in leaf_nodes]

        # Iteratively merge the closest pair of clusters until the desired number of clusters is reached
        while len(clusters) > n_clusters:
            # Find the closest pair of clusters
            min_distance = float('inf')
            closest_pair = None
            for i in range(len(clusters)):
                for j in range(i+1, len(clusters)):
                    distance = self._compute_cluster_distance(clusters[i], clusters[j])
                    if distance < min_distance:
                        min_distance = distance
                        closest_pair = (i, j)

            # Merge the closest pair of clusters
            clusters[closest_pair[0]].extend(clusters[closest_pair[1]])
            del clusters[closest_pair[1]]

        # Return the clusters
        return clusters

    def _get_leaf_nodes(self, node):
        # If the node is a leaf, return it
        if node.is_leaf:
            return [node]

        # If the node is not a leaf, get the leaf nodes from the children
        leaf_nodes = []
        for child in node.children:
            leaf_nodes.extend(self._get_leaf_nodes(child))

        return leaf_nodes

    def _compute_cluster_distance(self, cluster1, cluster2):
        # Compute the distance between two clusters as the average distance between their unions
        unions1 = [node.cf_vector.LS / node.cf_vector.n for node in cluster1]
        unions2 = [node.cf_vector.LS / node.cf_vector.n for node in cluster2]
        return sum(self._distance(union1, union2) for union1 in unions1 for union2 in unions2) / (len(unions1) * len(unions2))