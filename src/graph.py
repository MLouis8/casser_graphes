import networkx as nx
import matplotlib.pyplot as plt
import osmnx as ox
import kahip

def display_graph(graph):
    """
    Affichage matplotlib du graphe
    """
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_edges(graph, pos)
    edge_labels = nx.get_edge_attributes(graph, "weight")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_family='sans-serif')
    plt.show()

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
    G_ = add_nodes_and_edges_weights_and_relabel(remove_parallel_and_self_loops(G))
    G_ = G_.to_undirected()

    vwght = []
    adjncy = []
    xadj = [0]
    for node_id, data in G_.nodes.data():
        neighbors = [n for n in G_.neighbors(node_id)].sort()
        adjncy += neighbors
        xadj.append(xadj[-1]+len(neighbors))
        vwght.append(data['weight'])

    adjcwgt = []
    dict_edges_attributes = nx.get_edge_attributes(G_,'weight')
    for i in range(1,len(xadj)):
        for j in range(xadj[i-1],xadj[i]):
            try:
                adjcwgt.append(int(dict_edges_attributes[(i-1, adjncy[j], 0)]))
            except:
                adjcwgt.append(int(dict_edges_attributes[(adjncy[j], i-1, 0)]))
    
    return vwght, xadj, adjcwgt, adjncy

def add_fictive_node(G):
    # Identifier les arêtes parallèles et les stocker dans un dictionnaire
    parallel_edges = [(u,v,k) for u,v,k in G.edges if k !=0]
    
    # Créer un nouveau graphe pour stocker le résultat
    G_updated = G.copy()
    
    edges_weight = nx.get_edge_attributes(G,'weight')
    poids_renseigne = edges_weight != {}
    
    # Pour chaque ensemble d'arêtes parallèles, ajouter un nœud fictif et deux arêtes fictives
    for edge in parallel_edges:
        # Utiliser l'extrémité d'origine de l'ensemble d'arêtes comme position du nœud fictif
        u, v, k= edge[0], edge[1], edge[2]
        x_mid, y_mid = (G_updated.nodes[u]['x']+G_updated.nodes[v]['x'])/2, (G_updated.nodes[u]['y']+G_updated.nodes[v]['y'])/2
    
        # Créer un nœud fictif à l'extrémité d'origine de l'ensemble d'arêtes
        new_node = max(G_updated.nodes) + 1
        G_updated.add_node(new_node, x=x_mid, y=y_mid)

        G_updated.add_edge(u,new_node,0)
        G_updated.add_edge(new_node,v,0)
        if poids_renseigne :
            edges_weight[(u,new_node,0)] = int(edges_weight[(u,v,k)])
            edges_weight[(new_node,v,0)] = int(edges_weight[(u,v,k)])
            
        # Récupérer les attributs de l'arête source
        attributs_source = G.edges[(u,v,k)]
        attributs_source_2 = G.edges[(u,v,0)]
        
        # L'arête avec keys = 0 récupère les attributs de l'arrête secondaire (l'arête avec keys = 0)
        G_updated.edges[(u, v, 0)].update(attributs_source)
        G_updated.edges[(u, new_node, 0)].update(attributs_source_2)
        G_updated.edges[(new_node, v, 0)].update(attributs_source_2)
        
        # On retire l'arête parallèle
        G_updated.remove_edge(u, v,k)

    node_weights = {}
    for node in G_updated.nodes:
        node_weights[node] = 1

    nx.set_node_attributes(G_updated,node_weights,'weight')
    nx.set_edge_attributes(G_updated,edges_weight,'weight')
    
    return G_updated

def from_ox_to_coupe_fictive_node(G, nb_blocks, imbalance, mode):

    G_ = G.copy()
    
    # On retire les selfloops
    G_.remove_edges_from(nx.selfloop_edges(G_))

    # On ajoute les poids
    G_ = add_nodes_and_edges_weights_and_relabel(G_)
    
    # On ajoute les noeuds fictifs pour supprimer les parallel edges
    G_ = add_fictive_node(G_)
    
    # On convertit le graphe ox en graphe kahip
    kahip_G = from_nx_to_kahip_fictive(G_)

    vwgt,xadj,adjcwgt,adjncy = kahip_G[0], kahip_G[1], kahip_G[2], kahip_G[3]

    # On applique kahip
    edgecut, blocks = kahip.kaffpa(vwgt, xadj, adjcwgt, adjncy, nb_blocks, imbalance, 0, 0, mode)
    
    coupe_G = aretes_coupees(xadj, adjcwgt, adjncy, blocks)

    coupe_bis = []
    for coupe in coupe_G:
        coupe_bis.append((coupe[0],coupe[1],0))

    print(coupe_bis)
    print(edgecut)
    
    ec = ["r" if (u,v,k) in coupe_bis or (v,u,k) in coupe_bis else "gray" for u,v,k in G_.edges]
    nc = [blocks[n]+2 for n in G_.nodes]
    fig, ax = ox.plot_graph(
        G_, node_color=nc, node_edgecolor="k", node_size=5, edge_color=ec, edge_linewidth=1)
    
def from_nx_to_kahip_fictive(G):
    
    G_ = G.to_undirected()
    
    # Les noeuds
    nodes = [i for i in range(0,len(G_.nodes))]
    
    # Les poids des noeuds
    node_weights = nx.get_node_attributes(G_,'weight')
    vwght = [node_weights[i] for i in range(0,len(nodes))]

    # La liste d'adjacence et xadj
    adjncy = []
    xadj = [0]
    for i in nodes:
        voisins = [n for n in G_.neighbors(i)]
        # bien penser à trier la liste
        voisins.sort()
        adjncy += voisins
        xadj.append(xadj[-1]+len(voisins))
        
    # Les poids des arêtes
    adjcwgt = []
    dict_edges_attributes = nx.get_edge_attributes(G_,'weight')
    for i in range(1,len(xadj)):
        for j in range(xadj[i-1],xadj[i]):
            edge = (i-1,adjncy[j])
            # Si (u,v) n'est pas dans le dict nx, (v,u) l'est
            try:
                adjcwgt.append(dict_edges_attributes[(edge[0],edge[1],0)])
            except:
                adjcwgt.append(dict_edges_attributes[(edge[1],edge[0],0)])

    # On convertit les poids en entiers
    adjcwgt = list(map(int,adjcwgt))
    
    return vwght, xadj, adjcwgt, adjncy

# Fonction qui retourne le premier élément de la liste si l'argument est une liste, et renvoie l'argument sinon
def list_to_premier_element(l):
    if isinstance(l,list):
        return l[0]
    return l

# On essaie de faire un truc_propre
def from_ox_to_coupe_valuee(G,nb_blocks, imbalance, mode,val=None):

    def width(x):
        return float(x)
    
    def width_squared(x):
        return int(float(x)**2)        

    def no_valuation():
        return(1)

    if val is None:
        f = no_valuation
    elif val == 'width':
        f = width
    elif val == 'width_squared':
        f = width_squared
    else:
        return('Valuation inconnue. Les valuations disponibles sont None (pas de valuation), "width", et "width_squared"')

    dict_mapping_type_lanes = mapping_type_lanes(G)   
            
    mapping_edge_type = nx.get_edge_attributes(G, 'highway')

    for edge in mapping_edge_type.keys():
        mapping_edge_type[edge] = list_to_premier_element(mapping_edge_type[edge])

    mapping_edge_width = nx.get_edge_attributes(G, 'width')

    for edge in mapping_edge_width.keys():
        mapping_edge_width[edge] = replace_list_with_max(mapping_edge_width[edge])
    
    mapping_edge_lanes = nx.get_edge_attributes(G, 'lanes')

    for edge in mapping_edge_lanes.keys():
        mapping_edge_lanes[edge] = replace_list_with_max(mapping_edge_lanes[edge])

    for edge in G.edges:
        if edge not in mapping_edge_width.keys():
            if edge not in mapping_edge_lanes.keys():
                # Si on n'a ni le nb de voies, ni la largeur, on passe par le type de voie pour récupérer le nombre de voies
                mapping_edge_width[edge] = 5*dict_mapping_type_lanes[mapping_edge_type[edge]] #le facteur 5 est arbitraire
            else:
                mapping_edge_width[edge] = 5*mapping_edge_lanes[edge]

    G_ = G.copy()
    
    nx.set_edge_attributes(G_, mapping_edge_width,'width')

    dict_weights = mapping_edge_width.copy()

    for edge in dict_weights.keys():
        dict_weights[edge] = f(dict_weights[edge])

    nx.set_edge_attributes(G_,dict_weights,'weight')
    
    # On retire les selfloops
    G_.remove_edges_from(nx.selfloop_edges(G_))

    G_ = G_.to_undirected()
    
    # On ajoute les poids
    G_ = add_node_weights_and_relabel(G_)
    
    # On ajoute les noeuds fictifs pour supprimer les parallel edges
    G_ = add_fictive_node(G_)
    
    # On convertit le graphe ox en graphe kahip
    kahip_G = from_nx_to_kahip_fictive(G_)


    vwgt,xadj,adjcwgt,adjncy = kahip_G[0], kahip_G[1], kahip_G[2], kahip_G[3]

    print(len(vwgt))
    print(len(adjncy)/2)
    
    # On applique kahip
    edgecut, blocks = kahip.kaffpa(vwgt, xadj, adjcwgt, adjncy, nb_blocks, imbalance, 0, 0, mode)
    
    coupe_G = aretes_coupees(xadj, adjcwgt, adjncy, blocks)

    coupe_bis = []
    for coupe in coupe_G:
        coupe_bis.append((coupe[0],coupe[1],0))

    print(coupe_bis)
    print(edgecut)
    print(blocks) 
    
    ec = ["r" if (u,v,k) in coupe_bis or (v,u,k) in coupe_bis else "black" for u,v,k in G_.edges]
    nc = ['green' if blocks[n] == 0 else 'blue' for n in G_.nodes]
    fig, ax = ox.plot_graph(
        G_, node_color=nc, bgcolor='white', node_size=1, edge_color=ec, edge_linewidth=1)


# Fonction qui ajoute les attributs poids aux nodes à un graphe osmnx et qui relabelle les noeuds de 0 à n
def add_node_weights_and_relabel(G):

    G_ = G.copy()
    
    w_nodes = {}
    for node in list(G_.nodes):
        w_nodes[node] = 1 
    nx.set_node_attributes(G_,w_nodes, 'weight')

    # Obtenez une liste des nœuds triés par ordre croissant
    sorted_nodes = sorted(G_.nodes())

    # Créez un dictionnaire de correspondance entre les anciens nœuds et les nouveaux
    mapping = {old_node: new_node for new_node, old_node in enumerate(sorted_nodes)}

    G_ = nx.relabel_nodes(G_, mapping)

    return G_

# Fonction qui prétraite le graphe comme il faut (supression des selfloops, conversion en graphe non dirigé, ajout d'éventuels noeuds fictifs en cas d'arêtes parallèles)
def pretraitement(G):

    G_ = G.copy()
    G_.remove_edges_from(nx.selfloop_edges(G_))
    G_ = G_.to_undirected()
    G_ = add_node_weights_and_relabel(G_)
    G_ = add_fictive_node(G_)
    
    return G_

def pretraitement_bis(G):

    G_ = G.copy()
    G_.remove_edges_from(nx.selfloop_edges(G_))
    G_.to_undirected()
    G_ = add_node_weights_and_relabel(G_)
    G_ = add_fictive_node(G_)
    degrees = dict(G_.degree())
    for node in degrees.keys():
        if degrees[node] == 0:
            G_.remove_node(node)
    G_.to_undirected()
    return G_