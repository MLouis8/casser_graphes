# a executer a partir du repo Casser_graphe (chemin relatifs)

# Graphes sous format KaHIP, les arêtes sont valuées selon la fonction de coût précisée dans le nom
kp_paths = [
    "./data/old_costs/old_width.json",
    "./data/old_costs/nocost.json",
    "./data/old_costs/width.json",
    "./data/old_costs/widthsq.json",
    "./data/old_costs/widthmaxspeed.json",
    "./data/old_costs/widthnobridge.json",
    "./data/old_costs/widthnotunnel.json",
    "./data/old_costs/randomminmax.json",
    "./data/old_costs/randomdistr.json",
    "./data/costs/lanes.json",
    "./data/costs/lanessq.json",
    "./data/costs/lanesmaxspeed.json",
    "./data/costs/lanesnobridge.json"
]
costs_name = [
    "old_width",
    "nocost",
    "width",
    "widthsq",
    "widthmaxspeed",
    "widthnobridge",
    "widthnotunnel",
    "lanes",
    "lanessq",
    "lanesmaxspeed",
    "lanesnobridge"
]
# Graphes sous format graphml, W correspond à la deuxième fonction width (largeur inférée par lanes, highway et oneway) et L à la troisième (finale)
graphml_path = [
    "./data/Paris.graphml",
    "./data/ParisPreprocessedW.graphml",
    "./data/ParisPreprocessedL.graphml"
]
btw_path = "./data/betweenness_Paris.json"
freq_paths = [
    "./data/freqs/frequency_1000_cuts_Paris_01.json",
    "./data/freqs/frequency_1000_cuts_Paris_003.json",
    "./data/freqs/frequency_1000_cuts_Paris.json"
]
cut_paths_1 = [
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

cut_paths_2 = [
    "./data/cuts/1000_cuts_lanes_005.json",
    "./data/cuts/1000_cuts_lanessq_005.json",
    "./data/cuts/1000_cuts_lanesmaxspeed_005.json",
    "./data/cuts/1000_cuts_lanes_005nobridge.json",
    "./data/cuts/100_cuts_nocost_001.json",
    "./data/cuts/100_cuts_nocost_002.json",
    "./data/cuts/100_cuts_nocost_004.json",
    "./data/cuts/100_cuts_nocost_008.json",
    "./data/cuts/100_cuts_nocost_01.json",
    "./data/cuts/100_cuts_nocost_016.json",
    "./data/cuts/100_cuts_nocost_032.json"
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