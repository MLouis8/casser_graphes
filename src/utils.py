import networkx as nx
import matplotlib.pyplot as plt

def display_graph(G):
    """
    Affichage matplotlib du graphe
    """
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    plt.show()

def display_results(G, cut):
    """
    Affichage des resultats d'une coupe
    """
    print(f"Coupe: {cut[0]}")
    print(f"Differents blocks:")
    for i, block in enumerate(cut[1]):
        print(f"Block {i}: {block}")
    display_graph(G)

def kahip_to_nx(xadj: list[int], adjncy: list[int], vwgt: list[int], adjcwgt: list[int]):
    """
    Conversion du type KaHIP (adjacency) au type networkx.graph
    """
    G = nx.Graph()

    for i in range(len(vwgt)):
        G.add_node(i, weight=vwgt[i])

    aretes = []
    for i in range(1,len(xadj)):
        for j in range(xadj[i-1],xadj[i]):
            aretes.append((i-1,adjncy[j],adjcwgt[j]))
    G.add_weighted_edges_from(aretes)

    return G

def nx_to_kahip(G):
    """
    Conversion du type networkx.graph au type KaHIP (METIS)
    """
    G_ = G.copy()
    G_ = G_.to_undirected()

    vwght = []
    adjncy = []
    xadj = [0]
    for node_id, data in G_.nodes.data():
        neighbors = [n for n in G_.neighbors(node_id)]
        neighbors.sort()
        adjncy += neighbors
        xadj.append(xadj[-1] + len(neighbors))
        vwght.append(data['weight'])

    adjcwgt = []
    for i in range(1, len(xadj)):
        for j in range(xadj[i-1],xadj[i]):
            if (i-1, adjncy[j]) in nx.get_edge_attributes(G_,'weight'):
                adjcwgt.append(nx.get_edge_attributes(G_,'weight')[(i-1, adjncy[j])])
            else:
                adjcwgt.append(nx.get_edge_attributes(G_,'weight')[(adjncy[j], i-1)])

    
    return vwght, xadj, adjcwgt, adjncy

def replace_parallel_edges(G):
    """
    KaHIP ne suppporte pas les aretes paralleles, on les remplace donc
    par un noeud qui sert d'intermediaire pour une nouvelle arete.
    """
    parallel_edges = [(u,v,k) for u,v,k in G.edges if k !=0]
    
    edges_weight = nx.get_edge_attributes(G,'weight')
    
    for edge in parallel_edges:
        u, v, k = edge[0], edge[1], edge[2]
        x_mid = (G.nodes[u]['x'] + G.nodes[v]['x']) / 2
        y_mid = (G.nodes[u]['y'] + G.nodes[v]['y']) / 2
    
        new_node = max(G.nodes) + 1
        G.add_node(new_node, x=x_mid, y=y_mid)
        G.add_edge(u,new_node,0)
        G.add_edge(new_node,v,0)

        # Si les aretes sont values alors les nouvelles en heritent
        if edges_weight:
            edges_weight[(u,new_node,0)] = int(edges_weight[(u,v,k)])
            edges_weight[(new_node,v,0)] = int(edges_weight[(u,v,k)])
        
        # Pareil pour les attributs
        G.edges[(u, v, 0)].update(G.edges[(u,v,k)])
        G.edges[(u, new_node, 0)].update(G.edges[(u,v,0)])
        G.edges[(new_node, v, 0)].update(G.edges[(u,v,0)])
        
        # Sauf pour la longueur qui est divisee par deux
        # G.edges[(u, new_node, 0)]['']
        G.remove_edge(u, v,k)

    node_weights = {}
    for node in G.nodes:
        node_weights[node] = 1

    nx.set_node_attributes(G, node_weights, 'weight')
    nx.set_edge_attributes(G, edges_weight, 'weight')
    
def preprocessing(G):
    """
    Does all the required preprocessing and returns the preprocessed graph.
    """
    def add_node_weights_and_relabel(G):
        w_nodes = {}
        for node in list(G.nodes):
            w_nodes[node] = 1 
        nx.set_node_attributes(G, w_nodes, 'weight')
        sorted_nodes = sorted(G.nodes())
        mapping = {
            old_node: new_node for new_node, old_node 
                in enumerate(sorted_nodes)
        }
        G = nx.relabel_nodes(G, mapping)
    Gcopy = G.copy()
    Gcopy.remove_edges_from(nx.selfloop_edges(Gcopy))
    add_node_weights_and_relabel(Gcopy)
    Gcopy = replace_parallel_edges(Gcopy)
    Gcopy.remove_nodes_from(list(nx.isolates(Gcopy)))
    Gcopy.to_undirected()

def rebuilding():
    return
