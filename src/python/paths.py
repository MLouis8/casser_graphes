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
    "./data/costs/lanesnobridge.json",
    "./data/costs/randomminmax1-12.json",
    "./data/costs/randomlanedistr,json"
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
    "./data/ParisPreprocessedL.graphml",
    "./data/ParisPreprocessedBC.graphml"
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
    "./data/cuts/rddistr_1000_01.json",
    "./data/cuts/rddistr_1000_003.json",
    "./data/cuts/rddistr_1000_005.json",
]

cut_paths_2 = [
    "./data/cuts/lanes_1000_005.json",
    "./data/cuts/lanessq_1000_005.json",
    "./data/cuts/lanesmaxspeed_1000_005.json",
    "./data/cuts/lanesnobridge_1000_005.json",
    "./data/cuts/nocost_1000_000.json",
    "./data/cuts/nocost_1000_001.json",
    "./data/cuts/nocost_1000_002.json",
    "./data/cuts/nocost_1000_003.json",
    "./data/cuts/nocost_1000_004.json",
    "./data/cuts/nocost_1000_005.json",
    "./data/cuts/nocost_1000_008.json",
    "./data/cuts/nocost_1000_010.json",
    "./data/cuts/nocost_1000_012.json",
    "./data/cuts/nocost_1000_016.json",
    "./data/cuts/nocost_1000_024.json",
    "./data/cuts/nocost_1000_032.json",
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
    "./data/clusters/CTS_50000_nocost.json",
    "./data/clusters/CTS_50000_width.json",
    "./data/clusters/CTS_50000_widthsq.json",
    "./data/clusters/CTS_60000_widthsq.json",
    "./data/clusters/CTS_50000_widthmaxspeed.json",
    "./data/clusters/CTS_50000_widthnobridge.json",
    "./data/clusters/CTS_50000_widthnotunnel.json",
    "./data/clusters/CTS_50000_randomminmax.json",
]

clusters_paths_3 = [
    "./data/clusters/CTS_5000_lanes.json",
    "./data/clusters/CTS_7500_lanes.json",
    "./data/clusters/CTS_60000_lanessq.json",
    "./data/clusters/CTS_60000_lanesmaxspeed.json",
    "./data/clusters/CTS_70000_lanesmaxspeed.json",
    "./data/clusters/CTS_70000_lanesmaxspeed.json",
    "./data/clusters/CTS_40000_lanesnobridge.json",
    "./data/clusters/CTS_7500_lanesnobridge.json",
    "./data/clusters/CTS_10000_lanesnobridge.json",
    "./data/clusters/CTS_50000_lanesnobridge.json",
    "./data/clusters/CTS_60000_lanesnobridge.json",
]

rpaths = [
    "data/robust/nocost_graph_BC_5.json",
    "data/robust/nocost_graph_BC_10.json",
    "data/robust/nocost_graph_Freq_5.json",
    "data/robust/nocost_graph_Freq_10.json",
    "data/robust/nocost_graph_RD_5.json",
    "data/robust/nocost_graph_RD_10.json",
    "data/robust/nocost_graph_Deg_10.json",
]