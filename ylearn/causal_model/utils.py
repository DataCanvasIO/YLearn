import networkx as nx


def check_nodes(nodes=None, *S):
    S = filter(None, S)
    for i in S:
        for j in i:
            assert j in nodes


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
