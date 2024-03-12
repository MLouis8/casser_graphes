import networkx as nx
import osmnx as ox
import kahip
import json


class Graph:
    def __init__(
        self, vwgt=[], xadj=[], adjcwgt=[], adjncy=[], nx=None, json=None
    ) -> None:
        self._vertices_weight = vwgt
        self._xadjacency = xadj
        self._adjacency_weight = adjcwgt
        self._adjacency = adjncy

        self._edgecut = 0
        self._blocks = []

        self._nx = False

        if nx:
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
    def get_last_results(self):
        return (self._edgecut, self._blocks)
    
    def set_last_results(self, edgecut, blocks):
        self._edgecut, self._blocks = edgecut, blocks

    def set_from_nx(self, G):
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
        """
        Conversion du type KaHIP (adjacency) au type networkx.graph
        """
        G = nx.Graph()

        for i in range(self._sizeV):
            G.add_node(i, weight=self["vwgt"][i])

        aretes = []
        for i in range(1, len(self["xadj"])):
            for j in range(self["xadj"][i - 1], self["xadj"][i]):
                aretes.append((i - 1, self["adjncy"][j], self["adjcwgt"][j]))
        G.add_weighted_edges_from(aretes)

        return G

    def kaffpa_cut(
        self,
        nblocks,
        imbalance,
        suppress_output,
        seed,
        mode,
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
        self._edgecut, self._blocks = kahip.kaffpa(
            self["vwgt"],
            self["xadj"],
            self["adjcwgt"],
            self["adjncy"],
            nblocks,
            imbalance,
            suppress_output,
            seed,
            mode
        )

    def process_cut(self, weight=False):
        edges = []
        for i in range(1, len(self["xadj"])):
            for j in range(self["xadj"][i - 1], self["xadj"][i]):
                edges.append(
                    (i - 1, self["adjncy"][j], self["adjcwgt"][j] if weight else 0)
                )

        cut_edges = []
        for edge in edges:
            if self._blocks[edge[0]] != self._blocks[edge[1]]:
                if not edge in cut_edges:
                    cut_edges.append(edge)

        return cut_edges

    def display_city_cut(self, G_nx, savefig=False, filepath=None, show=True, ax=None, figsize=None):
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
            figsize=figsize
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

    def save_graph(self, filepath):
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

    def import_from_json(self, filepath):
        with open(filepath, "r") as read_file:
            data = json.load(read_file)
        self._vertices_weight = data["vwgt"]
        self._xadjacency = data["xadj"]
        self._adjacency_weight = data["adjcwgt"]
        self._adjacency = data["adjncy"]

    def compute_edge_betweenness(self):
        """Computes edge betweenness centrality and returns the dict assigning to each edge its betweenness"""
        if not self._nx:
            self._nx = self.to_nx()
        return nx.edge_betweenness_centrality(self._nx)

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
