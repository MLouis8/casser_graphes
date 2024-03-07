import networkx as nx
import osmnx as ox


class Graph:
    def __init__(self, vwght=[], xadj=[], adjcwgt=[], adjncy=[]) -> None:
        self._vertices_weight = vwght
        self._xadjacency = xadj
        self._adjacency_weight = adjcwgt
        self._adjacency = adjncy
        self._vertices = []

    def __getitem__(self, key: str) -> list[int]:
        match key:
            case "vwght":
                return self._vertices_weight
            case "xadj":
                return self._xadjacency
            case "adjcwgt":
                return self._adjacency_weight
            case "adjncy":
                return self._adjacency
            case _:
                raise ValueError("Possible keys are: 'vwght'/'xadj'/'adjcwgt'/'adjncy'")

    def set_from_nx(self, G):
        """
        Conversion du type networkx.graph au type KaHIP (METIS)
        """
        G_ = G.copy().to_undirected()

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
        """
        Conversion du type KaHIP (adjacency) au type networkx.graph
        """
        G = nx.Graph()

        for i in range(len(self["vwgt"])):
            G.add_node(i, weight=self["vwgt"][i])

        aretes = []
        for i in range(1, len(self["xadj"])):
            for j in range(self["xadj"][i - 1], self["xadj"][i]):
                aretes.append((i - 1, self["adjncy"][j], self["adjcwgt"][j]))
        G.add_weighted_edges_from(aretes)

        return G

    def process_cut(self, comp_cnx, weight=False):
        edges = []
        for i in range(1, len(self["xadj"])):
            for j in range(self["xadj"][i - 1], self["xadj"][i]):
                edges.append(
                    (i - 1, self["adjncy"][j], self["adjcwgt"][j] if weight else 0)
                )

        cut_edges = []
        for edge in edges:
            if comp_cnx[edge[0]] != comp_cnx[edge[1]]:
                if not edge in cut_edges:
                    cut_edges.append(edge)

        return cut_edges

    def display_city_cut(self, G_nx, cut):
        p_cut = self.process_cut(cut[1])
        ec = [
            "r" if (u, v, k) in p_cut or (v, u, k) in p_cut else "black"
            for u, v, k in G_nx.edges
        ]
        nc = ["green" if cut[1][n] == 0 else "blue" for n in G_nx.nodes]

        ox.plot_graph(
            G_nx,
            node_color=nc,
            bgcolor="white",
            node_size=1,
            edge_color=ec,
            edge_linewidth=1,
        )

    def display_cut_results(self, cut):
        p_cut = self.process_cut(cut[1])
        print(f"Coupe: C = [{p_cut}] de taille {cut[0]}")
        print(f"\nAvec la repartition en blocks suivante:")
        for i in range(len(vwght)):
            block = []
            for j in cut[1]:
                if i == j:
                    block.append(j)
            print(f"Dans le block {i} il y a les noeuds: {block}")
