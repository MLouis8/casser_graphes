from multiprocessing import Pool
import itertools
import json
from Graph import Graph
from paths import graphml_path, kp_paths, robust_paths_directed, effective_res_paths
import random as rd

import matplotlib.pyplot as plt
import networkx as nx


def chunks(l, n):
    """Divide a list of nodes `l` in `n` chunks"""
    l_c = iter(l)
    while 1:
        x = tuple(itertools.islice(l_c, n))
        if not x:
            return
        yield x


def betweenness_centrality_parallel(G, processes=None):
    """Parallel betweenness centrality  function"""
    p = Pool(processes=processes)
    node_divisor = len(p._pool) * 4
    node_chunks = list(chunks(G.nodes(), G.order() // node_divisor))
    num_chunks = len(node_chunks)
    bt_sc = p.starmap(
        nx.betweenness_centrality_subset,
        zip(
            [G] * num_chunks,
            node_chunks,
            [list(G)] * num_chunks,
            [True] * num_chunks,
            ['weight'] * num_chunks,
        ),
    )

    # Reduce the partial solutions
    bt_c = bt_sc[0]
    for bt in bt_sc[1:]:
        for n in bt:
            bt_c[n] += bt[n]
    return bt_c


def main():
    G = Graph(json=kp_paths[9])
    G_nx = G.to_nx(directed=True)
    weigths = nx.get_edge_attributes(G_nx, "weight")
    new_weights = {}
    for k, v in weigths.items():
        new_weights[k] = int(v)
    nx.set_edge_attributes(G_nx, new_weights, "weight")
    for i in range(0, 51):
        print(f"computing {i}th freq")
        bc = betweenness_centrality_parallel(G_nx)
        if i > 0:
            cut_union = []
            seen_seeds = []
            for _ in range(1000):
                seed = rd.randint(0, 1044642763)
                while seed in seen_seeds:
                    seed = rd.randint(0, 1044642763)
                seen_seeds.append(seed)
                G.kaffpa_cut(2, 0.05, 0, seed, 2)
                cut_union += G.process_cut()
            frequencies = {}
            for edge in cut_union:
                if edge in frequencies:
                    frequencies[edge] += 1
                else:
                    frequencies[edge] = 1
            chosen_edge = max(frequencies, key=frequencies.get)
            G.remove_edge(chosen_edge)
            G_nx.remove_edge(chosen_edge[0], chosen_edge[1])
        else:
            chosen_edge = None
        with open("data/robust/directed/lanes_graphdir_freq2_50.json", "r") as rfile:
            data = json.load(rfile)
        bcsave = {}
        for k, v in bc.items():
            bcsave[str(k)] = v
        data.append((str(chosen_edge), bcsave))
        with open("data/robust/directed/lanes_graphdir_freq2_50.json", "w") as wfile:
            json.dump(data, wfile)

main()