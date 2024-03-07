import networkx as nx
import osmnx as ox

def replace_parallel_edges(G):
    """
    KaHIP ne suppporte pas les aretes paralleles, on les remplace donc
    par un noeud qui sert d'intermediaire pour une nouvelle arete.
    """
    parallel_edges = [(u, v, k) for u, v, k in G.edges if k != 0]

    edges_weight = nx.get_edge_attributes(G, "weight")

    for edge in parallel_edges:
        u, v, k = edge[0], edge[1], edge[2]
        x_mid = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2
        y_mid = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2

        new_node = max(G.nodes) + 1
        G.add_node(new_node, x=x_mid, y=y_mid)
        G.add_edge(u, new_node, 0)
        G.add_edge(new_node, v, 0)

        # Si les aretes sont valuees alors les nouvelles en heritent
        if edges_weight:
            edges_weight[(u, new_node, 0)] = int(edges_weight[(u, v, k)])
            edges_weight[(new_node, v, 0)] = int(edges_weight[(u, v, k)])

        # Pareil pour les attributs
        G.edges[(u, v, 0)].update(G.edges[(u, v, k)])
        G.edges[(u, new_node, 0)].update(G.edges[(u, v, 0)])
        G.edges[(new_node, v, 0)].update(G.edges[(u, v, 0)])

        # Sauf pour la longueur qui est divisee par deux
        G.edges[(u, new_node, 0)]["length"] = G.edges[(u, v, 0)]["length"] / 2
        G.edges[(new_node, v, 0)]["length"] = G.edges[(u, v, 0)]["length"] / 2

        G.remove_edge(u, v, k)

    node_weights = {}
    for node in G.nodes:
        node_weights[node] = 1

    nx.set_node_attributes(G, node_weights, "weight")
    nx.set_edge_attributes(G, edges_weight, "weight")


def preprocessing(G, val: str = "no valuation"):
    """
    Does all the required preprocessing in place and returns the preprocessed graph.
    """
    pp1 = lambda x: x[0] if isinstance(x, list) else x
    pp2 = lambda x: float(max(x) if isinstance(x, list) else x) if x else 0

    def add_node_weights_and_relabel(G):
        w_nodes = {}
        for node in list(G.nodes):
            w_nodes[node] = 1
        nx.set_node_attributes(G, w_nodes, "weight")
        sorted_nodes = sorted(G.nodes())
        mapping = {old_node: new_node for new_node, old_node in enumerate(sorted_nodes)}
        G = nx.relabel_nodes(G, mapping)

    def map_type_lanes(G):
        """Function mapping lanes to their type, used for inferring width attribute."""
        gdfs = ox.graph_to_gdfs(G)
        gdf_l_h = gdfs[1].loc[~gdfs[1]["lanes"].isna() & ~gdfs[1]["highway"].isna()]

        gdf_l_h.loc[:, "lanes"] = gdf_l_h["lanes"].apply(pp2)
        gdf_l_h.loc[:, "highway"] = gdf_l_h["highway"].apply(pp1)
        edges = gdf_l_h[["highway", "lanes"]]
        hist_data = edges.groupby(["highway", "lanes"]).size().reset_index(name="count")
        max_count_indices = hist_data.groupby("highway")["count"].idxmax()
        max_count = hist_data.loc[max_count_indices]
        map = {max_count["highway"][i]: max_count["lanes"][i] for i in max_count.index}
        # convert highway to 2 lanes
        gdf_h = gdfs[1]["highway"].apply(pp1)
        for highway in gdf_h.unique():
            if highway not in map.keys():
                map[highway] = 2

        return map

    match val:
        case "width":
            val = lambda x: float(x)
        case "squared width":
            val = lambda x: int(float(x) ** 2)
        case _:
            val = lambda _: 1
    t_l_map = map_type_lanes(G)
    edge_type = nx.get_edge_attributes(G, "highway")
    edge_type = dict((k, pp1(v)) for k, v in edge_type.items())

    edge_width = nx.get_edge_attributes(G, "width")
    edge_width = dict((k, pp2(v)) for k, v in edge_width.items())

    edge_lanes = nx.get_edge_attributes(G, "lanes")
    edge_lanes = dict((k, pp2(v)) for k, v in edge_lanes.items())

    for edge in G.edges:
        if not edge in edge_width.keys():
            if not edge in edge_lanes.keys():
                # on infere avec le type de voie
                edge_width[edge] = (
                    4 * t_l_map[edge_type[edge]]
                )  # le facteur 4 correspond a la moyenne des largeurs des rues de Paris
            else:
                edge_width[edge] = 4 * edge_lanes[edge]
    nx.set_edge_attributes(G, edge_width, "width")
    for edge in edge_width.keys():
        edge_width[edge] = val(edge_width[edge])
    nx.set_edge_attributes(G, edge_width, "weight")

    G.remove_edges_from(nx.selfloop_edges(G))
    add_node_weights_and_relabel(G)
    replace_parallel_edges(G)
    G.to_undirected()
