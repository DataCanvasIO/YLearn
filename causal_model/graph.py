from causal_model import prob
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
        self.edges = []
        for k, v in causation.items():
            for para in v:
                self.edges.append((para, k))
        self.dag = nx.DiGraph()
        self.dag.add_edges_from(self.edges)

    @property
    def is_dag(self):
        # TODO: determin if the graph is a DAG, try tr(e^{W\circledot W}-d)=0
        return nx.is_directed_acyclic_graph(self.dag)

    def add_nodes(self, node_list, create=False):
        pass

    def add_edges_from(self, edge_list, create=False):
        """Add edges to the causal graph.

        Parameters
        ----------
        edge_list : list
            every element of the list contains two elements, the first for
            the parent
        create : bool
            return a new graph if set as True
        """
        self.dag.add_edges_from(edge_list)
        for edge in edge_list:
            self.causation[edge[1]].append(edge[0])

    def add_edge(self, i, j, create=False):
        self.dag.add_edge(i, j)
        self.causation[j].append(i)

    def remove_nodes(self, node_list, create=False):
        pass

    def remove_edges_from(self, edge_list, create=False):
        self.dag.remove_edges_from(edge_list)
        for edge in edge_list:
            self.causation[edge[1]].remove(edge[0])

    def remove_edge(self, i, j, create=False):
        self.dag.remove_edges(i, j)
        self.causation[j].remove(i)

    def to_adj_matrix(self):
        W = nx.to_numpy_matrix(self.dag)
        return W

    def to_adj_list(self):
        pass

    @property
    def c_components(self):
        """Return the C-component set of the graph.

        Returns
        ----------
        c : set
            the C-component set of graph
        """
        c = None
        return c

    def ancestors(self, x):
        """Return the ancestors of all nodes in x.

        Parameters
        ----------
        x : set
            a set of nodes in the graph

        Returns
        ----------
        an : set
            ancestors of nodes in x of the graph
        """
        pass

    @property
    def observed_part(self):
        """Return the observed subgraph of the graph.

        Returns
        ----------
        ob_graph : CausalGraph
            the observed part of the graph
        """
        pass

    @property
    def topo_order(self):
        """Retrun the topological order of the nodes in the observed graph

        Returns
        ----------
        topological_order : generator
            nodes in the topological order
        """
        pass

    def build_ancestor_graph(self, y, create=True):
        """Construct the ancestor graph of nodes in y.

        Parameters
        ----------
        y : set
            the set of variables for the ancestor graph to be constructed with
        create : bool
            return a new graph if set as Ture

        Returns
        ----------
        an_graph : CausalGraph
            ancestor graph of the node y
        """
        pass
    
    def build_sub_graph(self, subset):
        """Construct the subgraph with the nodes in subset"""
        pass

    def remove_incoming_edges(self, x, create=True):
        """remove incoming edges of all nodes in x.

        Parameters
        ----------
        x : set
        create : bool
            return a new graph if set as Ture

        Returns
        ----------
        modified_graph : CausalGraph
            subgraph of the graph without all incoming edges of nodes in x
        """
        pass

    def remove_outcoming_edges(self, x, create=True):
        """remove outcoming edges of all nodes in x.

        Parameters
        ----------
        x : set
        create : bool

        Returns
        ----------
        modified_graph : CausalGraph
            subgraph of the graph without all outcoming edges of nodes in x
        """
        pass
