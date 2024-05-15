scc1 = []
    for attack in bcdata:
        scc = nx.strongly_connected_components(G_nx)
        scc1.append(max(scc, key=len))
        try:
            edge = eval(attack[0])
            G_nx.remove_edge(edge[0], edge[1])
        except:
            print(attack[0])
    scc2 = []
    G_nx = begin.copy()
    for attack in freqdata:
        scc = nx.strongly_connected_components(G_nx)
        scc2.append(max(scc, key=len))
        try:
            edge = eval(attack[0])
            G_nx.remove_edge(edge[0], edge[1])
            print("no")
        except:
            G_nx.remove_edge(edge[1], edge[0])
    scc3 = []
    G_nx = begin.copy()
    for attack in degdata:
        scc = nx.strongly_connected_components(G_nx)
        scc3.append(max(scc, key=len))
        try:
            edge = eval(attack[0])
            G_nx.remove_edge(edge[0], edge[1])
        except:
            print(attack[0])
    visualize_attack_scores([scc1, scc2, scc3], ["bc", "freq", "deg"], "data/notweighted_basestrat_scc.pdf", False, "Scc evolution")