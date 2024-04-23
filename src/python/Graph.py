import networkx as nx
import osmnx as ox
import numpy as np

# # import kahip  # to comment if ARM, uncomment to cut
import json
from typing import Optional, Any
from typ import EdgeDict
from math import log, exp


class Graph:
    def __init__(
        self,
        vwgt: list[int] = [],
        xadj: list[int] = [],
        adjcwgt: list[int] = [],
        adjncy: list[int] = [],
        nx: nx.Graph | None = None,
        json: str | None = None,
        bc: EdgeDict | None = None,
    ) -> None:
        self._vertices_weight = vwgt
        self._xadjacency = xadj
        self._adjacency_weight = adjcwgt
        self._adjacency = adjncy

        self._edgecut = 0  # 2
        self._blocks: list[int] = []  # [0, 0, 1, 1, 0]

        self._nx = nx
        self._bc = bc
        self._old_bc = False
        self._cf_bc: None | EdgeDict = None
        self._avg_dist: None | float = None
        self._adj_spectrum: None | np.ndarray = None
        self._spectral_gap: None | float = None
        self._spectral_rad: None | float = None
        self._nat_co: None | float = None

        if nx:
            if not json:
                self.set_from_nx(nx)
            self._nx = nx
        if json:
            self.import_from_json(json)

        self._sizeV = len(self._vertices_weight)
        self._sizeE = len(self._adjacency_weight) // 2

    def __getitem__(self, key: str) -> list[int]:
        match key:
            case "vwgt":
                return self._vertices_weight
            case "xadj":
                return self._xadjacency
            case "adjcwgt":
                return self._adjacency_weight
            case "adjncy":
                return self._adjacency
            case _:
                raise ValueError("Possible keys are: 'vwgt'/'xadj'/'adjcwgt'/'adjncy'")

    @property
    def get_last_results(self) -> tuple[int, list[int]]:
        return (self._edgecut, self._blocks)

    def set_last_results(self, edgecut: int, blocks: list[int]) -> None:
        self._edgecut, self._blocks = edgecut, blocks

    def get_neighbors(self, x: int) -> list[int]:
        return [self["adjncy"][n] for n in range(self["xadj"][x + 1] - self["xadj"][x])]

    def set_from_nx(self, G: nx.Graph) -> None:
        """Conversion du type networkx.graph au type KaHIP (METIS)"""

        G_ = G.to_undirected()

        self._adjacency = []
        self._xadjacency = [0]
        node_weights = nx.get_node_attributes(G_, "weight")
        self._vertices_weight = [node_weights[i] for i in range(len(G_.nodes))]

        for node in range(len(G_.nodes)):
            neighbors = [n for n in G_.neighbors(node)]
            neighbors.sort()
            self._adjacency += neighbors
            self._xadjacency.append(self._xadjacency[-1] + len(neighbors))

        self._adjacency_weight = []
        dict_edges_attributes = nx.get_edge_attributes(G_, "weight")
        for i in range(1, len(self._xadjacency)):
            for j in range(self._xadjacency[i - 1], self._xadjacency[i]):
                try:
                    self._adjacency_weight.append(
                        dict_edges_attributes[(i - 1, self._adjacency[j], 0)]
                    )
                except:
                    self._adjacency_weight.append(
                        dict_edges_attributes[(self._adjacency[j], i - 1, 0)]
                    )

    def to_nx(self):
        """Conversion du type KaHIP (adjacency) au type networkx.graph"""
        G = nx.Graph()

        for i in range(self._sizeV):
            G.add_node(i, weight=self["vwgt"][i])

        aretes = []
        for i in range(1, len(self["xadj"])):
            for j in range(self["xadj"][i - 1], self["xadj"][i]):
                aretes.append((i - 1, self["adjncy"][j], self["adjcwgt"][j]))
        G.add_weighted_edges_from(aretes)

        return G

    def remove_edge(self, edge: tuple[int, int]) -> None:
        if self._nx:
            self._nx.remove_edge(edge[0], edge[1])
        if not self._old_bc:
            self._old_bc = True
        n1, n2 = edge if edge[0] < edge[1] else (edge[1], edge[0])
        new_xadj = []
        for i in range(self._sizeV + 1):
            if i <= n1:
                new_xadj.append(self["xadj"][i])
            elif i <= n2:
                new_xadj.append(self["xadj"][i] - 1)
            else:
                new_xadj.append(self["xadj"][i] - 2)
        id1, id2 = 0, 0
        for j in range(self["xadj"][n1], self["xadj"][n1 + 1]):
            if self["adjncy"][j] == n2:
                id1 = j
                break
        for j in range(self["xadj"][n2], self["xadj"][n2 + 1]):
            if self["adjncy"][j] == n1:
                id2 = j
                break
        self._adjacency.pop(id1)
        self._adjacency.pop(id2 - 1)
        self._xadjacency = new_xadj
        self._adjacency_weight.pop(id1)
        self._adjacency_weight.pop(id2 - 1)

    def closer_than_k_edges(
        self, e1: tuple[int, int], e2: tuple[int, int], k: int
    ) -> bool:
        """Return whether d(e1, e2) <= k"""
        for i in range(4):
            closer = self.closer_k_nodes(e1[i // 2], e2[i % 2], k - 1)
            if closer:
                return True
        return False

    def closer_k_nodes(self, n1: int, n2: int, k: int) -> bool:
        if n1 == n2:
            return True
        if k == 0:
            return False
        neighbors = self.get_neighbors(n1)
        for neighbor in neighbors:
            if self.closer_k_nodes(neighbor, n2, k - 1):
                return True
        return False

    def isolating_cut(self, nodes, seed, imb=0.03):
        new_weight = (self._sizeV - len(nodes)) / len(nodes)
        rest = self._sizeV - new_weight
        for node in nodes:
            self["vwght"][node] = new_weight
        self["vwght"][-1] += rest
        self.kaffpa_cut(2, imb, 0, seed, 2)

    def zone_cut(self, nodes, seed, imb=0.03):
        # first we verify that the nodes define a zone
        # then we update the nodes weight
        for node in nodes:
            pass
        edges = []
        # then we update links weight
        for edge in edges:
            pass

    def kaffpa_cut(
        self,
        nblocks: int,
        imbalance: float,
        suppress_output: int,
        seed: int,
        mode: int,
    ):
        """
        Alias for kaffpa cut

        set mode:
        - FAST         = 0
        - ECO          = 1
        - STRONG       = 2
        - FASTSOCIAL   = 3
        - ECOSOCIAL    = 4
        - STRONGSOCIAL = 5

        Strong should be used if quality is
        paramount, eco if you need a good tradeoff between partition qual-
        ity and running time, and fast if partitioning speed is in your focus.
        Configurations with a social in their name should be used for social
        networks and web graphs.
        """
        return
        # self._edgecut, self._blocks = kahip.kaffpa(
        #     self["vwgt"],
        #     self["xadj"],
        #     self["adjcwgt"],
        #     self["adjncy"],
        #     nblocks,
        #     imbalance,
        #     suppress_output,
        #     seed,
        #     mode,
        # )

    def process_cut(self) -> list[tuple[int, int]]:
        edges = []
        for i in range(1, len(self["xadj"])):
            for j in range(self["xadj"][i - 1], self["xadj"][i]):
                edges.append((i - 1, self["adjncy"][j]))
        cut_edges = []
        for edge in edges:
            if self._blocks[edge[0]] != self._blocks[edge[1]]:
                if not edge in cut_edges and not (edge[1], edge[0]) in cut_edges:
                    cut_edges.append(edge)
        return cut_edges

    def display_city_cut(
        self,
        G_nx: nx.Graph,
        savefig: bool = False,
        filepath: Optional[str] = None,
        show: bool = True,
        ax=None,
        figsize: Optional[tuple[int, int]] = None,
    ):
        p_cut = self.process_cut()
        ec = [
            "r" if (u, v, k) in p_cut or (v, u, k) in p_cut else "black"
            for u, v, k in G_nx.edges
        ]
        nc = ["green" if self._blocks[n] == 0 else "purple" for n in G_nx.nodes]
        show = False if savefig else show
        return ox.plot_graph(
            G_nx,
            node_color=nc,
            bgcolor="white",
            node_size=0.1,
            edge_color=ec,
            edge_linewidth=0.1,
            save=savefig,
            filepath=filepath,
            show=show,
            dpi=500,
            ax=ax,
            figsize=figsize,
        )

    def display_last_cut_results(self):
        p_cut = self.process_cut()
        print(f"Coupe: C = [{p_cut}] de taille {self._edgecut}")
        print(f"\nAvec la repartition en blocks suivante:")
        for i in range(self._edgecut):
            block = []
            for node, j in enumerate(self._blocks):
                if i == j:
                    block.append(node)
            print(f"Dans le block {i} il y a les noeuds: {block}")

    def save_graph(self, filepath: str) -> None:
        data = {
            "vwgt": self._vertices_weight,
            "xadj": self._xadjacency,
            "adjcwgt": self._adjacency_weight,
            "adjncy": self._adjacency,
        }
        with open(filepath, "w") as write_file:
            json.dump(
                data, write_file, indent=4, separators=(", ", ": "), sort_keys=True
            )

    def import_from_json(self, filepath: str) -> None:
        with open(filepath, "r") as read_file:
            data = json.load(read_file)
        self._vertices_weight = data["vwgt"]
        self._xadjacency = data["xadj"]
        self._adjacency_weight = data["adjcwgt"]
        self._adjacency = data["adjncy"]

    def get_connected_components(self) -> Any:
        """Get connected components from KaHIP graph using NetworkX"""
        if not self._nx:
            self._nx = self.to_nx()
        cut = self.process_cut()
        G_copy = self._nx.copy()
        G_copy.remove_edges_from(cut)
        return nx.connected_components(G_copy)

    def rmv_small_cc_from_cut(self, treshold: int) -> None:
        """ "
        Removes small connected components from the cut
        (the cost of cutting these small components)
        (in place)
        Set the processed cut as last_result
        """
        if not self._nx:
            self._nx = self.to_nx()
        size, blocks = self.get_last_results()
        weights = nx.get_edge_attributes(self._nx, "weight")
        for component in self.get_connected_components(self._nx):
            if len(component) < treshold:
                cut = self.process_cut()
                for node in component:
                    # on retire le coût des arêtes concernées
                    for edge in cut:
                        if node == edge[0] or node == edge[1]:
                            size -= weights[edge]
                    # et on change de bloc les elements du component
                    blocks[node] = 1 - blocks[node]
        self.set_last_results(size, blocks)

    @property
    def get_size_biggest_cc(self):
        if not self._nx:
            self._nx = self.to_nx()
        return len(sorted(nx.connected_components(self._nx), key=len, reverse=True)[0])

    def get_edge_bc(self, new: bool = False) -> EdgeDict:
        if not self._nx:
            self._nx = self.to_nx()
        if self._old_bc or new:
            self._bc = nx.edge_betweenness_centrality(self._nx)
            self._old_bc = False
        return self._bc

    @property
    def get_avg_edge_bc(self) -> float:
        return np.mean(list(self.get_edge_bc().values()))

    @property
    def get_edge_cf_bc(self) -> EdgeDict:
        if not self._nx:
            self._nx = self.to_nx()
        return nx.edge_current_flow_betweenness_centrality(self._nx)

    @property
    def get_avg_edge_cf_bc(self) -> float:
        return np.mean(list(self.get_edge_cf_bc.values()))

    @property
    def get_avg_dist(self) -> float:
        if not self._nx:
            self._nx = self.to_nx()
        return nx.average_shortest_path_length(self._nx)

    def cpt_adj_spectrum(self) -> None:
        if not self._nx:
            self._nx = self.to_nx()
        self._adj_spectrum = nx.adjacency_spectrum(self._nx)

    @property
    def get_spectral_radius(self) -> float:
        if not self._adj_spectrum:
            self.cpt_adj_spectrum()
        return np.max(self._adj_spectrum)

    @property
    def get_spectral_gap(self) -> float:
        if not self._adj_spectrum:
            self.cpt_adj_spectrum()
        max1, max2 = 0, 0
        for eigen in self._adj_spectrum:
            if eigen > max2:
                if eigen > max1:
                    max2 = max1
                    max1 = eigen
                else:
                    max2 = eigen
        return max1 - max2

    @property
    def get_natural_co(self) -> float:
        if not self._nx:
            self._nx = self.to_nx()
        return log(nx.subgraph_centrality(self._nx) / self._sizeV)
