import networkx as nx


class CausalGraph:
    """Causal Graph.

    Attributes
    ----------
    causation : dic
        data structure of the causal graph where values are parents of the
        corresponding keys
    observed_var : list
    unobserved_var : list
    DG : nx.DiGraph
        graph represented by the networkx package
    edges : list
        list of all edges

    Methods
    ----------
    is_dag()
        Determine whether the constructed graph is a DAG.
    add_edges_from(edge_list)
        Add all edges in the edge_list to the graph.
    add_edge(i, j)
        Add an edge between nodes i and j.
    remove_edges_from(edge_list)
        Remove all edges in the edge_list in the graph.
    remove_edge(i, j)
        Remove the edge between nodes i and j.
    to_adj_matrix()
        Return the adjacency matrix.
    to_adj_list()
        Return the adjacency list.
    c_components()
        Return the C-components of the graph.
    observed_part()
        Return the observed part of the graph, including observed nodes and
        edges between them.
    """

    def __init__(self, causation, observed):
        self.causation = causation
        self.observed_var = set(observed)
        self.unobserved_var = set(causation.keys()) - self.observed_var
        init_edges = []
        for k, v in causation.items():
            for para in v:
                init_edges.append((para, k))
        self.DG = nx.DiGraph()
        self.DG.add_edges_from(init_edges)
        self.edges = self.DG.edges

    def is_dag(self):
        # TODO: determin if the graph is a DAG, try tr(e^{W\circledot W}-d)=0
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

    def c_components(self):
        """Return the C-component set of the graph.

        Parameters
        ----------
        """
        pass
    
    def ancestors(self, x):
        """Return the ancestors of node x.

        Parameters
        ----------
        x : str
            a node in the graph
        """
        pass

    def observed_part(self):
        pass
    
    def ancestor_graph(self, y):
        pass
