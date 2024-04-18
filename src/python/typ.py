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
# Robustness List: list of size n (the number of attacks) containing 3-tuples of:
    # - the cut edge
    # - the EBC after the removel of the edge
    # - the size of the biggest connex component
RobustList = list[tuple[Edge, EdgeDict, int]]