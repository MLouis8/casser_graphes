## Type Aliases ##
KCuts = tuple[int, list[int]]  # cuts under KaHIP format
EdgeDict = dict[tuple[int, int], int]  # common edge dict
EdgeDictStr = dict[str, int]  # edge dict after json import
Cuts = dict[
    str, list[tuple[int, int]]
]  # cuts after post processing, the name of the cut maps to the list of edges cut
Classes = list[list[str]]  # list of list of names of cuts
Cut = list[tuple[int, int]]