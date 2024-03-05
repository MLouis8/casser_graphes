import networkx as nx
import kahip
import matplotlib.pyplot as plt
import osmnx as ox
import numpy as np
from sklearn.linear_model import LinearRegression

# Fonction qui permet d'afficher un graphe nx
def display_graph(graph):
    """
    Display a graph using spring_layout and node labels
    """

    # position all the nodes
    pos = nx.spring_layout(graph)

    # draw graph element, by element
    # nodes
    nx.draw_networkx_nodes(graph, pos)

    # edges
    nx.draw_networkx_edges(graph, pos)

    # weights
    edge_labels = nx.get_edge_attributes(graph, "weight")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels)
    
    # labels
    nx.draw_networkx_labels(graph, pos, font_size=10, font_family='sans-serif')

    plt.show()

# Fonction qui convertit un graphe kahip en graphe nx
def from_kahip_to_nx(xadj,adjncy,vwgt,adjcwgt,display=False):
    # La fonction prend les mêmes arguments que KaHIP. On a seulement rajouté l'option "display" qui affiche le graphe si on en a envie
    
    # On initialise le graphe
    G = nx.Graph()

    # On ajoute les noeuds (pondérés)
    for i in range(len(vwgt)):
        G.add_node(i, weight=vwgt[i])

    # On ajoute les arêtes (pondérées)
    aretes = []
    for i in range(1,len(xadj)):
        for j in range(xadj[i-1],xadj[i]):
            aretes.append((i-1,adjncy[j],adjcwgt[j]))
    G.add_weighted_edges_from(aretes)

    # On affiche le graphe si l'option display a été choisie
    if display:
        display_graph(G)

    return G

# Fonction qui convertit un grahe nx en graphe kahip
def from_nx_to_kahip(G):
    
    G_ = add_nodes_and_edges_weights_and_relabel(remove_parallel_and_self_loops(G))
    G_ = G_.to_undirected()
        
    # Les poids des noeuds
    node_weights = nx.get_node_attributes(G_,'weight')
    vwght = [node_weights[i] for i in range(len(G_.nodes))]

    # La liste d'adjacence et xadj
    adjncy = []
    xadj = [0]
    for i in G_.nodes:
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
                adjcwgt.append(int(dict_edges_attributes[(edge[0],edge[1],0)]))
            except:
                adjcwgt.append(int(dict_edges_attributes[(edge[1],edge[0],0)]))
    
    return((vwght,xadj,adjcwgt,adjncy))


# Fonction qui, étant donnés un graphe kahip et la liste de ses composantes connexes après la coupe, retourne les arêtes coupées et le poids de la coupe
def aretes_coupees(xadj, adjcwgt, adjncy, comp_cnx):

    # On initialise la liste qui va récupérée les arêtes coupées
    aretes_coupees = []


    # On récupère les arêtes du graphe
    aretes = []
    for i in range(1,len(xadj)):
        for j in range(xadj[i-1],xadj[i]):
            aretes.append((i-1,adjncy[j],adjcwgt[j]))
            
    # Une arete à ses extrémités dans des composantes connexes différentes ssi elle appartient à la coupe
    for arete in aretes:
        if comp_cnx[arete[0]] != comp_cnx[arete[1]]:
            if (arete[1],arete[0],arete[2]) not in aretes_coupees :
                aretes_coupees.append(arete)

    return aretes_coupees


# Fonction qui prend en argument un graphe kahip et ses aretes_coupees et renvoie un graphe nx avec les arêtes coupees
def graphe_apres_coupe(xadj, adjncy,vwgt,adjcwgt, aretes_coupees, display = False):

    G = from_kahip_to_nx(xadj, adjncy,vwgt,adjcwgt)

    for arete in aretes_coupees:
        G.remove_edge(arete[0],arete[1])
        
    if display:
        display_graph(G)

    return(G)

# Fonction qui prend en entrée un graphe nx et renvoie le graphe après la coupe
def pipeline_from_nx(G, nblocks, imbalance, supress_output, seed, mode, display = True):

    # On convertit le graphe nx en graphe KaHIP
    liste_kahip = from_nx_to_kahip(G)
    vwgt, xadj, adjcwgt, adjncy = liste_kahip[0], liste_kahip[1], liste_kahip[2], liste_kahip[3]

    # On applique kahip au graphe
    nb_blocks, comp_cnx = kahip.kaffpa(vwgt, xadj, adjcwgt, adjncy, nblocks, imbalance, supress_output, seed, mode)

    coupe = aretes_coupees(xadj, adjcwgt, adjncy, comp_cnx)[0]

    graphe_apres_coupe(xadj, adjncy,vwgt,adjcwgt, coupe, display)

# Fonction qui retire les selfloops et parallel edges d'un graphe nx, utile car kahip ne prend en argument que des graphes sans parallel edges et sans self loops
def remove_parallel_and_self_loops(G):
    """deprecated"""
    G_ = G.copy()
    # Supression des self loops
    G_.remove_edges_from(nx.selfloop_edges(G_))

    # Supression des parallel edges
    parallel_edges = [(u,v) for u,v,k in G_.edges if k != 0]
    G_.remove_edges_from(parallel_edges)

    return G_


# Fonction qui ajoute les attributs poids aux nodes et edges à un graphe osmnx et qui relabelle les noeuds de 0 à n
def add_nodes_and_edges_weights_and_relabel(G):
    
    G_ = G.copy()
    w = {}
    for edge in G_.edges:
        w[edge] = 1
    nx.set_edge_attributes(G_, w, 'weight')

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


# Prend en argument un graphe ox, relabelle les noeuds et ajoute les poids 
def clean_ox(G):
    return add_nodes_and_edges_weights_and_relabel(remove_parallel_and_self_loops(G))

# On écrit une pipeline qui prend en argument un graphe ox original, le nombre de blocs que l'on veut, le déséquilibre
# et un mode et affiche le graphe avec les arêtes coupées et les composantes connexes
def from_ox_to_coupe(G, nb_blocks, imbalance=0.1, mode=0):

    # On ajoute les poids et relabelle les noeuds
    G_ = clean_ox(G)
    
    # On convertit le graphe ox en graphe kahip
    kahip_G = from_nx_to_kahip(G_)

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

# Fonction qui prend en argument les blocks et le graphe ox de départ non touché, et qui renvoie les arêtes
# coupées avec les labels des noeuds du graphe ox de départ
def mapping_inverse(coupe,G_base):
    
    G_ = G_base.copy()

    # On retire les selfloops
    G_.remove_edges_from(nx.selfloop_edges(G_))

    G_ = G_.to_undirected()
    
    # On ajoute les noeuds fictifs pour supprimer les parallel edges
    G_ = add_fictive_node(G_)

    kahip_G = from_nx_to_kahip(G_)
    vwght,xadj,adjcwgt,adjncy = kahip_G[0], kahip_G[1], kahip_G[2], kahip_G[3]
    
    coupe_base = []
    
    sorted_nodes = sorted(G_.nodes())
    
    mapping = {new_node: old_node for new_node, old_node in enumerate(sorted_nodes)}

    aretes_coupees_ = aretes_coupees(xadj, adjcwgt, adjncy, coupe)
    
    for arete in aretes_coupees_:
        coupe_base.append((mapping[arete[0]],mapping[arete[1]],0))
    
    return(coupe_base)

# On affiche le nom des arêtes coupées 
def nom_aretes_coupees(coupe_base,G_base):
    for arete in coupe_base:
        try:
            print(nx.get_edge_attributes(G_base,'name')[arete])
        except:
            print(nx.get_edge_attributes(G_base,'name')[(arete[1],arete[0],0)])


# Fonction qui applique KaHIP au graphe et affiche le nom des arêtes coupees
def from_nx_to_aretes_coupees_base(G_base, nblocks=2, imbalance=0.1, seed=0, mode=0):
    
    kahip_G = from_nx_to_kahip(G_base)
    
    vwght,xadj,adjcwgt,adjncy = kahip_G[0], kahip_G[1], kahip_G[2], kahip_G[3]

    _, comp_cnx = kahip.kaffpa(vwght, xadj, adjcwgt, adjncy, nblocks, imbalance, 0, seed, mode)

    coupe_base = mapping_inverse(comp_cnx,G_base)
    
    nom_aretes_coupees(coupe_base,G_base)


# Prend en argument un graphe et un edge attribute et renvoie le graphe de l'edge_betweenness en fonction de l'attribut choisi
# Ce module prend quelques minutes à s'éxécuter sur des gros graphes
def plot_betweenness(G,attribut):
    # Calcul de l'edge_betweenness_centrality
    edge_betweenness = nx.edge_betweenness_centrality(G)

    # Récupération du nombre de voies pour chaque arête
    attribute_data = ox.graph_to_gdfs(G, nodes=False)[[attribut, 'geometry']]
    num_attribute_values = []

    # Traiter les cas où l'attribut est une liste
    for attribute, _ in attribute_data.itertuples(index=False):
        # Si le nombre de voies est une liste, prenez la valeur maximale
        if isinstance(attribute, list):
            num_attribute_values.append(max(map(float, attribute)))
        elif not np.nan is attribute:
            num_attribute_values.append(float(attribute))
        else :
            num_attribute_values.append(0)

    # Création d'une liste de tuples (edge_betweenness, nombre_de_voies)
    data = [(edge_betweenness[edge], num_attribute) for edge, num_attribute in zip(edge_betweenness, num_attribute_values)]

    # On ne garde que les edges pour lesquels l'attribut choisi est renseigné
    data = [edge_attribute for edge_attribute in data if edge_attribute[1] != 0]
    
    # Séparation des données en deux listes pour le tracé
    centrality_values, num_attribute_values = zip(*data)

    # Tracé du graphique
    plt.scatter(num_attribute_values, centrality_values, alpha=0.5)
    plt.title(f'Edge Betweenness Centrality en fonction de {attribut}')
    plt.xlabel(f'{attribut}')
    plt.ylabel('Edge Betweenness Centrality')
    plt.show()

# Prend en argument une liste de strings avec des noms de villes enregistrées et renvoie leur coûts
def cout_coupes(liste_ville):
    for ville in liste_ville:
        G = ox.load_graphml(filepath = f'{ville.replace(", ","_").replace(" ","_")}.graphml')
        kahip_G = from_nx_to_kahip(G)
        edge_cut = (kahip.kaffpa(kahip_G[0],kahip_G[1],kahip_G[2],kahip_G[3],2,0.1,0,0,0))[0]
        print(ville, edge_cut)


def replace_list_with_max(value):
    if isinstance(value, list):
        return max(map(float, value))
    try:
        return float(value)
    except:
        return 0

# Fonction qui prend en argument un graphe et retourne la régression linéaire de la largeur de la voie en fonction du nombre de voies
# Utile car la width n'est pas souvent renseignée alors que le nombre de lanes l'est plus souvent
def reg_lin(G):
    _, gdf_edges = ox.graph_to_gdfs(G)
    gdf_lanes_and_width = gdf_edges.loc[(~gdf_edges['width'].isna()) & (~gdf_edges['lanes'].isna())]
    gdf_lanes_and_width['width'] = gdf_lanes_and_width['width'].apply(replace_list_with_max)
    gdf_lanes_and_width['lanes'] = gdf_lanes_and_width['lanes'].apply(replace_list_with_max)
    x = np.array(gdf_lanes_and_width['lanes'])
    x = x.reshape(-1,1)
    y = gdf_lanes_and_width['width']
    model = LinearRegression()
    model.fit(x,y)
    return model

# Fonction qui prend en argument un graphe et retourne une copie du graphe avec les width données par le modèle de régression linéaire
# 
def add_extrapolated_width(G):
    model = reg_lin(G)
    dict_lanes = nx.get_edge_attributes(G, 'lanes')
    dict_width = nx.get_edge_attributes(G, 'width')
    for edge in dict_lanes.keys():
        dict_lanes[edge] = replace_list_with_max(dict_lanes[edge])
    moy_width = np.mean(list(dict_lanes.values()))
    for edge in G.edges:
        if edge not in dict_width.keys():
            if edge in dict_lanes.keys():
                dict_width[edge] = model.predict([[dict_lanes[edge]]])[0]
            else:
                dict_width[edge] = moy_width
    G_ = G
    nx.set_edge_attributes(G_, dict_width, 'width')
    return G_

# Fonction qui retourne une liste de villes 
def villes_exemple():
    return ['Milan, Italie','Napoli, Italie','New_York','Shanghai', 'Sidney', 'Washington_D.C._USA', 'Varsovie, Pologne','Chicago, USA','Bruxelles, Belgique','Bombay','Barcelone, Espagne','Bordeaux']

def betweenness_Paris():
    #add json content
    print('attention contenu enleve')
    return()

def plot_betweenness_Paris(G,attribut):
    # Calcul de l'edge_betweenness_centrality
    edge_betweenness = betweenness_Paris()

    # Récupération du nombre de voies pour chaque arête
    attribute_data = ox.graph_to_gdfs(G, nodes=False)[[attribut, 'geometry']]
    num_attribute_values = []

    # Traiter les cas où l'attribut est une liste
    for attribute, _ in attribute_data.itertuples(index=False):
        # Si le nombre de voies est une liste, prenez la valeur maximale
        if isinstance(attribute, list):
            num_attribute_values.append(max(map(float, attribute)))
        elif not np.nan is attribute:
            num_attribute_values.append(float(attribute))
        else :
            num_attribute_values.append(0)

    # Création d'une liste de tuples (edge_betweenness, nombre_de_voies)
    data = [(edge_betweenness[edge], num_attribute) for edge, num_attribute in zip(edge_betweenness, num_attribute_values)]

    # On ne garde que les edges pour lesquels l'attribut choisi est renseigné
    data = [edge_attribute for edge_attribute in data if edge_attribute[1] != 0]
    
    # Séparation des données en deux listes pour le tracé
    centrality_values, num_attribute_values = zip(*data)

    # Tracé du graphique
    plt.scatter(num_attribute_values, centrality_values, alpha=0.5)
    plt.title(f'Edge Betweenness Centrality en fonction de {attribut}')
    plt.xlabel(f'{attribut}')
    plt.ylabel('Edge Betweenness Centrality')
    plt.show()

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

# Fonction qui retourne le mapping entre le type de rue et le nombre de voies
def mapping_type_lanes(G):

    _, gdf_edges = ox.graph_to_gdfs(G)
    
    # On récupère toutes les rues ayant un nb de voies renseigné et un type de rue renseignés
    gdf_edges_lanes = gdf_edges.loc[(~gdf_edges['lanes'].isna())&(~gdf_edges['highway'].isna())]

    gdf_edges_lanes['lanes'] = gdf_edges_lanes['lanes'].apply(replace_list_with_max)
    gdf_edges_lanes['highway'] = gdf_edges_lanes['highway'].apply(list_to_premier_element)
    
    edges = gdf_edges_lanes[['highway','lanes']]

    # Grouper par type de highway et compter le nombre d'occurrences pour chaque nombre de voies
    hist_data = edges.groupby(['highway', 'lanes']).size().reset_index(name='count')

    max_count_indices = hist_data.groupby('highway')['count'].idxmax()
    mapping_type_lanes = hist_data.loc[max_count_indices]

    dict_mapping = {mapping_type_lanes['highway'][i]:mapping_type_lanes['lanes'][i] for i in mapping_type_lanes.index}

    gdf_highway = gdf_edges['highway'].apply(list_to_premier_element)
    for highway in gdf_highway.unique():
        if highway not in dict_mapping.keys():
            dict_mapping[highway] = 2
        
    return(dict_mapping)


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

def from_ox_to_coupe_valuee_maxspeed(G,nb_blocks, imbalance, mode):

    G_= G.copy()
    ox.add_edge_speeds(G_)

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

    dict_maxspeed = nx.get_edge_attributes(G_,'speed_kph')
    
    nx.set_edge_attributes(G_, mapping_edge_width,'width')

    dict_weights = mapping_edge_width.copy()
    
    for edge in dict_weights.keys():
        try :
            dict_weights[edge] = dict_weights[edge] + int(dict_maxspeed[edge])**2
        except :
            dict_weights[edge] = dict_weights[edge] + int(dict_maxspeed[(edge[1],edge[0],edge[2])])**2

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
    
    ec = ["r" if (u,v,k) in coupe_bis or (v,u,k) in coupe_bis else "gray" for u,v,k in G_.edges]
    nc = [blocks[n]+2 for n in G_.nodes]
    fig, ax = ox.plot_graph(
        G_, node_color=nc, node_edgecolor="k", node_size=5, edge_color=ec, edge_linewidth=1)

def betweenness_Paris_tol_4():
    print('attention contenu enleve')
    return

def coupe_betweenness_Paris_tol_4(nb_blocks,imbalance,mode):

    G = ox.load_graphml(filepath = 'Paris_Paris_tol_4.graphml')

    G = add_node_weights_and_relabel(G)
    
    betweenness = betweenness_Paris_tol_4()
    
    for edge in list(betweenness.keys()):
        betweenness[edge] = int(betweenness[edge]*1000000)
    
    G = G.to_undirected()
    
    nx.set_edge_attributes(G,betweenness,'weight')
    
    # On convertit le graphe ox en graphe kahip
    kahip_G = from_nx_to_kahip_fictive(G)


    vwgt,xadj,adjcwgt,adjncy = kahip_G[0], kahip_G[1], kahip_G[2], kahip_G[3]

    print(f'Nombre de noeuds: {len(vwgt)}')
    print(f'Nombre d\'arêtes: {len(adjncy)//2}')
    
    # On applique kahip
    edgecut, blocks = kahip.kaffpa(vwgt, xadj, adjcwgt, adjncy, nb_blocks, imbalance, 0, 0, mode)
    
    coupe_G = aretes_coupees(xadj, adjcwgt, adjncy, blocks)

    coupe_bis = []
    for coupe in coupe_G:
        coupe_bis.append((coupe[0],coupe[1],0))

    print(coupe_bis)
    print(edgecut)
    print(blocks) 
    
    ec = ["r" if (u,v,k) in coupe_bis or (v,u,k) in coupe_bis else "black" for u,v,k in G.edges]
    nc = ['green' if blocks[n] == 0 else 'blue' for n in G.nodes]
    fig, ax = ox.plot_graph(
        G, node_color=nc, bgcolor='white', node_size=1, edge_color=ec, edge_linewidth=1
    )

# Ce module interdit les arêtes coupées par la première coupe.
def from_ox_to_coupe_valuee_interdite(G,nb_blocks, imbalance, mode,val=None):

    def width(x):
        return float(x)
    
    def width_squared(x):
        return int(float(x)**2)        

    def no_valuation(x):
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
    
    # On applique kahip
    edgecut, blocks = kahip.kaffpa(vwgt, xadj, adjcwgt, adjncy, nb_blocks, imbalance, 0, 0, mode)
    
    coupe_G = aretes_coupees(xadj, adjcwgt, adjncy, blocks)

    # On interdit (en pénalisant très fortement) les arêtes qui ont déjà été coupées
    for arete in coupe_G:
        dict_weights[(arete[0],arete[1],0)] = 1000000
        dict_weights[(arete[1],arete[0],0)] = 1000000
        
    nx.set_edge_attributes(G_,dict_weights,'weight')
    
    # On reconvertit le graphe ox en graphe kahip
    kahip_G = from_nx_to_kahip_fictive(G_)


    vwgt,xadj,adjcwgt,adjncy = kahip_G[0], kahip_G[1], kahip_G[2], kahip_G[3]

    print(len(vwgt))
    print(len(adjncy)//2)
    
    # On reapplique kahip
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
        G_, node_color=nc, bgcolor='white', node_size=1, edge_color=ec, edge_linewidth=1
    )

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

def from_ox_to_coupe_valuee_seed(G,nb_blocks, imbalance, mode, seed, val=None):

    def width(x):
        return float(x)
    
    def width_squared(x):
        return int(float(x)**2)        

    def no_valuation(x):
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
    edgecut, blocks = kahip.kaffpa(vwgt, xadj, adjcwgt, adjncy, nb_blocks, imbalance, 0, seed, mode)
    
    coupe_G = aretes_coupees(xadj, adjcwgt, adjncy, blocks)

    coupe_bis = []
    for coupe in coupe_G:
        coupe_bis.append((coupe[0],coupe[1],0))

    return edgecut
