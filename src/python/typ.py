from typing import Any

## Type Aliases ##
Edge = tuple[int, int]
KCut = tuple[int, list[int]]  # cuts under KaHIP format
EdgeDict = dict[Edge, float]  # common edge dict
EdgeDict3 = dict[tuple[int, int, int], int]  # common edge dict
EdgeDictStr = dict[str, int]  # edge dict after json import
Cuts = dict[
    str, list[Edge]
]  # cuts after post processing, the name of the cut maps to the list of edges cut
Classes = list[list[str]]  # list of list of names of cuts
Cut = list[Edge]
# Robustness Dictionary, it can contain:
#   "edges"         -> the ordered list of edges cut
#   "avg bc"        -> the ol. of average edge betweenness centralities
#   "avg cf bc"     -> the ol. of average edge current flow bcs
#   "avg dist"      -> the ol. of average distances
#   "spectral gap"  -> the ol. of spectral gaps
#   "spectral rad"  -> the ol. of spectral radius
#   "nat co"        -> the ol. of natural connectivity
RobustnessDict = dict[str, list[Any]]