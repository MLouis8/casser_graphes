import networkx as nx
import osmnx as ox
from Graph import Graph
import numpy as np
from colour import Color

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
            val = lambda x: int(x)
        case "squared width":
            val = lambda x: int(x ** 2)
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

# Resultats observes lors de l'execution
# Just after importation, we have : 
# 94783 edges
# 70263 nodes
# After consolidation, we have : 
# 59060 edges
# 40547 nodes
# After projection, we have : 
# 59060 edges
# 40547 nodes

def init_city_graph(filepath):
    G = ox.graph_from_place('Paris, Paris, France', network_type="drive", buffer_dist=350,simplify=False,retain_all=True,clean_periphery=False,truncate_by_edge=False)
    G_Paris = ox.project_graph(G, to_crs='epsg:2154') ## pour le mettre dans le même référentiel que les données de Paris

    print('Just after importation, we have : ')
    print(str(len(G.edges())) + ' edges')
    print(str(len(G.nodes()))+ ' nodes')
    G2 = ox.consolidate_intersections(G_Paris, rebuild_graph=True, tolerance=4, dead_ends=True)
    print('After consolidation, we have : ')
    print(str(len(G2.edges())) + ' edges')
    print(str(len(G2.nodes()))+ ' nodes')
    G_out = ox.project_graph(G2, to_crs='epsg:4326')
    print('After projection, we have : ')
    print(str(len(G_out.edges())) + ' edges')
    print(str(len(G_out.nodes()))+ ' nodes')
    ox.save_graphml(G_out, filepath=filepath)

# init_city_graph("./data/Paris.graphml")

def prepare_instance(filename):
    filepath_graph = "./data/"+filename+".graphml"
    filepath_kahip = "./data/"+filename+".json"
    print(f"Loading instance {filepath_graph}")
    G_nx = ox.load_graphml(filepath_graph)
    print(f"preprocessing the graph...")
    preprocessing(G_nx, val="width")
    print(f"Conversion into KaHIP format...")
    G_kp = Graph(nx=G_nx)
    G_kp.save_graph(filepath_kahip)

def basic_stats_edges(dictionary, g_size=46761, nb_cuts=1000):
    most_cut = max(dictionary, key=dictionary.get)
    less_cut = min(dictionary, key=dictionary.get)
    values = list(dictionary.values())
    mean = np.mean(values)
    std = np.std(values)

    print("Here some basic stats on this set of cut(s):")
    print(f"The most cut edge ({most_cut}) has been cut {dictionary[most_cut]} times")
    print(f"The less cut edge has been cut {dictionary[less_cut]} times")
    print(f"We have a mean of {mean}")
    print(f"We have an std of {std}")
    print(f"With {nb_cuts} cuts we have cut {len(dictionary.keys())} different edges over {g_size}")

def basic_stats_cuts(cuts, nb_cuts=1000):
    nb_edges_cut = cuts.values()[:, 0]
    best_cut = min(nb_edges_cut)
    worst_cut = max(nb_edges_cut)
    mean = np.mean(nb_edges_cut)
    std = np.std(nb_edges_cut)
    nb_best_cut = nb_cuts.count(best_cut)
    
    print("Here some basic stats on the set of cuts:")
    print(f"The best cut cuts {best_cut} edges")
    print(f"It appears {nb_best_cut} out of {nb_cuts} meaning a frequency of {nb_best_cut/nb_cuts}")
    print(f"The worst cut cuts {worst_cut} edges")
    print(f"For {nb_cuts} cuts we have a mean of {mean} cut edges")
    print(f"And a std of {std}")

    print
def display_freq(G_kp, G_nx, f, savefig=False, filepath=None, show=True, ax=None, figsize=None):
    def colorize2(u, v):
        if (u, v) in f:
            if f[(u,v)] > 500:
                return "#d80606"
            if f[(u,v)] > 450:
                return "#d83200"
            elif f[(u,v)] > 400:
                return "#d84a00"
            elif f[(u,v)] > 350:
                return "#d65d00"
            elif f[(u,v)] > 300:
                return "#d46e00"
            elif f[(u,v)] > 250:
                return "#d07d00"
            elif f[(u,v)] > 200:
                return "#cc8c00"
            elif f[(u,v)] > 175:
                return "#c79a00"
            elif f[(u,v)] > 150:
                return "#c1a700"
            elif f[(u,v)] > 125:
                return "#bab400"
            elif f[(u,v)] > 100:
                return "#b3c100"
            elif f[(u,v)] > 75:
                return "#abcd00" 
            elif f[(u,v)] > 50:
                return "#a2d928"
            elif f[(u,v)] > 25:
                return "#99e442"
            elif f[(u,v)] > 10:
                return "#8eef59"
            elif f[(u,v)] > 0:
                return "#83fa70"
        else:
            return '#54545460'
    def colorize(u, v):
        if (u, v) in f:
            if f[(u,v)] > 400:
                return 'tab:brown'
            elif f[(u,v)] > 300:
                return 'tab:purple'
            elif f[(u,v)] > 200:
                return 'tab:red'
            elif f[(u,v)] > 100:
                return 'tab:orange'
            elif f[(u,v)] > 50:
                return 'y'
            else:
                return 'g'
        else:
            return '#54545460'
    edge_color = [colorize(u, v) for u, v, _ in G_nx.edges]
    print("edges colorized, starting display...")
    show = False if savefig else show
    return ox.plot_graph(
        G_nx,
        bgcolor="white",
        node_size=1,
        edge_color=edge_color,
        edge_linewidth=1,
        save=savefig,
        filepath=filepath,
        show=show,
        dpi=1024,
        ax=ax,
        figsize=figsize,
        node_color="#54545460",
        edge_alpha=None
    )
