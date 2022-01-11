import networkx as nx


class CausalGraph:
    """Causal Graph.

    Attributes
    ----------
    causation : dic
        data structure of the causal graph where values are parents of the
            corresponding keys
    DG : nx.DiGraph
        graph represented by the networkx package
    edges : list
        list of all edges

    Methods
    ----------
    is_dag()
    add_edges_from()
    add_edge()
    remove_edges_from()
    remove_edge()
    to_adj_matrix()
    to_adj_list()
    """

    def __init__(self, causation):
        self.causation = causation
        init_edges = []
        for k, v in causation.items():
            for para in v:
                init_edges.append((para, k))
        self.DG = nx.DiGraph()
        self.DG.add_edges_from(init_edges)
        self.edges = self.DG.edges

    def is_dag(self):
        # TODO: determin if the graph is a DAG, can use tr(e^{W\circledot W}-d)=0
        return nx.is_directed_acyclic_graph(self.DG)

    def add_edges_from(self, edge_list):
        """Add edges to the causal graph.

        Parameters
        ----------
        edge_list : list
            every element of the list contains two elements, the first for
            the parent
        """
        self.DG.add_edges_from(edge_list)
        for edge in edge_list:
            self.causation[edge[1]].append(edge[0])

    def add_edge(self, i, j):
        self.DG.add_edge(i, j)
        self.causation[j].append(i)

    def remove_edges_from(self, edge_list):
        self.DG.remove_edges_from(edge_list)
        for edge in edge_list:
            self.causation[edge[1]].remove(edge[0])

    def remove_edge(self, i, j):
        self.DG.remove_edges(i, j)
        self.causation[j].remove(i)

    def to_adj_matrix(self):
        W = nx.to_numpy_matrix(self.DG)
        return W

    def to_adj_list(self):
        pass
