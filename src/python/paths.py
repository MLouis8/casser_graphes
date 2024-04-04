# a executer a partir du repo Casser_graphe (chemin relatifs)
kp_paths = [
    "./data/costs/old_width.json",
    "./data/costs/nocost.json",
    "./data/costs/width.json",
    "./data/costs/widthsq.json",
    "./data/costs/widthmaxspeed.json",
    "./data/costs/widthnobridge.json",
    "./data/costs/widthnotunnel.json",
    "./data/costs/randomminmax.json",
    "./data/costs/randomdistr.json",
]
costs_name = [
    "old_width",
    "nocost",
    "width",
    "widthsq",
    "widthmaxspeed",
    "widthnobridge",
    "widthnotunnel"
]
graphml_path = [
    "./data/Paris.graphml",
    "./data/ParisPreprocessedW.graphml"
]
btw_path = "./data/betweenness_Paris.json"
freq_paths = [
    "./data/freqs/frequency_1000_cuts_Paris_01.json",
    "./data/freqs/frequency_1000_cuts_Paris_003.json",
    "./data/freqs/frequency_1000_cuts_Paris.json"
]
cut_paths = [
    # 0
    "./data/cuts/1000_cuts_Paris_01.json",
    "./data/cuts/1000_cuts_Paris_003.json",
    "./data/cuts/1000_cuts_Paris.json",
    # 3
    "./data/cuts/nocost_1000_01.json",
    "./data/cuts/nocost_1000_003.json",
    "./data/cuts/nocost_1000_005.json",
    # 6
    "./data/cuts/width_1000_01.json",
    "./data/cuts/width_1000_003.json",
    "./data/cuts/width_1000_005.json",
    # 9
    "./data/cuts/widthsq_1000_01.json",
    "./data/cuts/widthsq_1000_003.json",
    "./data/cuts/widthsq_1000_005.json",
    # 12
    "./data/cuts/widthmaxspeed_1000_01.json",
    "./data/cuts/widthmaxspeed_1000_003.json",
    "./data/cuts/widthmaxspeed_1000_005.json",
    # 15
    "./data/cuts/widthnobridge_1000_01.json",
    "./data/cuts/widthnobridge_1000_003.json",
    "./data/cuts/widthnobridge_1000_005.json",
    # 18
    "./data/cuts/widthnotunnel_1000_01.json",
    "./data/cuts/widthnotunnel_1000_003.json",
    "./data/cuts/widthnotunnel_1000_005.json",
    # 21
    "./data/cuts/rdminmax_1000_01.json",
    "./data/cuts/rdminmax_1000_003.json",
    "./data/cuts/rdminmax_1000_005.json",
    # 24
    "./data/cuts/rddditr_1000_01.json",
    "./data/cuts/rddistr_1000_003.json",
    "./data/cuts/rddistr_1000_005.json",
]
clusters_paths_1 = [
    "./data/clusters/cluster_sum_003.json",
    "./data/clusters/cluster_inter_01.json",
    "./data/clusters/cluster_inter_003.json",
    "./data/clusters/cluster_t_70000.json",
    "./data/clusters/cluster_t_50000.json",
    "./data/clusters/cluster_t_30000.json",
    "./data/clusters/cluster_t_10000.json"
]

clusters_paths_2 = [
    "./data/clusters/CTS_50000nocost.json",
    "./data/clusters/CTS_50000width.json",
    "./data/clusters/CTS_50000widthsq.json",
    "./data/clusters/CTS_50000widthmaxspeed.json",
    "./data/clusters/CTS_50000widthnobridge.json",
    "./data/clusters/CTS_50000widthnotunnel.json",
    "./data/clusters/CTS_50000randomminmax.json",
    "./data/clusters/CTS_50000randomdistr.json",
]