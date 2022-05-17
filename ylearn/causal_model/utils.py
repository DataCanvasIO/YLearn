from re import I
import networkx as nx

from copy import deepcopy
from itertools import combinations


def ancestors_of_iter(g, x):
    """Return the ancestors of all nodes in x.

    Parameters
    ----------
    x : set of str
        a set of nodes in the graph

    Returns
    ----------
    set of str
        Ancestors of nodes x of the graph
    """
    an = set()
    x = {x} if isinstance(x, str) else x
    
    for node in x:
        an.add(node)
        try:
            an.update(nx.ancestors(g, node))
        except Exception:
            pass

    return an    


def descendents_of_iter(g, x):
    """Return the descendents of all nodes in x.

    Parameters
    ----------
    x : set of str
        a set of nodes in the graph

    Returns
    ----------
    set of str
        Descendents of nodes x of the graph
    """
    des = set()
    x = {x} if isinstance(x, str) else x
    
    for node in x:
        des.add(node)
        try:
            des.update(nx.descendants(g, node))
        except Exception:
            pass

    return des

def remove_ingo_edges(graph, incoming=True, *S):
    """Remove incoming or outgoing edges for nodes of S in the graph.

    Parameters
    ----------
    graph : nx.DiGraph, optional
        
    incoming : bool, optional
        If True, then all incoming edges of nodes in S will be removed,
        otherwise remove all outgoing edges of nodes in S, by default True

    Returns
    -------
    nx.DiGraph, optional
        A deepcopy of graph where the incoming edges or outgoing edges of nodes
        in S are removed.
    """
    g = deepcopy(graph)
    S = filter(None, S)
    
    for i in S:
        i = {i} if isinstance(i, str) else i
        check_nodes(g.nodes, i)
        for node in i:
            if incoming:
                g.remove_edges_from([(p, node) for p in g.predecessors(node)])
            else:
                g.remove_edges_from([(node, c) for c in g.successors(node)])

    return g

def check_nodes(nodes=None, *S):
    """Check if nodes contained in S are present in nodes.

    Parameters
    ----------
    nodes : list, optional
        A list of nodes which should include all nodes in S, by default None
    """
    S = filter(None, S)
    for i in S:
        i = {i} if isinstance(i, str) else i
        for j in i:
            assert j in nodes, f'The node {j} is not in all avaliable nodes {nodes}'


def check_ancestors_chain(dag, node, U):
    an = nx.ancestors(dag, node)

    if len(an) == 0:
        return True
    elif U in an:
        return False
    else:
        for n in an:
            if not check_ancestors_chain(dag, n, U):
                return False

        return True


def powerset(iterable):
    """Return the power set of the iterable.

    Parameters
    ----------
    iterable : container
        Can be a set or a list.

    Returns
    ----------
    list
        The list of power set.
    """
    # TODO: return a generator instead of a list
    s = list(iterable)
    power_set = []
    for i in range(len(s) + 1):
        for j in combinations(s, i):
            power_set.append(set(j))
    return power_set


class IdentificationError(Exception):
    def __init__(self, info):
        super().__init__(self)
        self.info = info

    def __str__(self):
        return self.info
