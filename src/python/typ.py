## Type Aliases ##
Edge = tuple[int, int]
KCut = tuple[int, list[int]]  # cuts under KaHIP format
EdgeDict = dict[Edge, int]  # common edge dict
EdgeDict3 = dict[tuple[int, int, int], int]  # common edge dict
EdgeDictStr = dict[str, int]  # edge dict after json import
Cuts = dict[
    str, list[Edge]
]  # cuts after post processing, the name of the cut maps to the list of edges cut
Classes = list[list[str]]  # list of list of names of cuts
Cut = list[Edge]