from sys import ps1
import causal_model
import networkx as nx
import copy

from causal_model import prob
from estimator_model.estimation_learner.meta_learner import SLearner


class CausalGraph:
    """Causal Graph.

    Attributes
    ----------
    causation : dic
        data structure of the causal graph where values are parents of the
        corresponding keys
    observed_var : list
    unobserved_var : list
    dag : nx.DiGraph
        graph represented by the networkx package

    Methods
    ----------
    prob()
        Return the Prob object of the CausalGraph.
    is_dag()
        Determine whether the constructed graph is a DAG.
    add_nodes(node_list, new)
        If not new, add all nodes in the node_list to the current
        CausalGraph, else new a new graph and add nodes.
    add_edges_from(edge_list, new)
        Add all edges in the edge_list to the graph.
    add_edge(i, j, new)
        Add an edge between nodes i and j.
    remove_edge(i, j, new)
        Remove the edge between nodes i and j.
    remove_edges_from(edge_list, new)
        Remove all edges in the edge_list in the graph.
    remove_nodes(node_list, new)
        Remove all nodes in the node_list. If new, do this in a new
        CausalGraph.
    to_adj_matrix()
        Return the adjacency matrix.
    to_adj_list()
        Return the adjacency list.
    c_components()
        Return the C-components of the graph.
    ancestors(y)
        Return ancestors of y.
    observed_part()
        Return the observed part of the graph, including observed nodes and
        edges between them.
    topo_order()
        Return a generator of the nodes in the topological order.
    build_sub_graph(subset)
        Return a new CausalGraph as the subgraph of graph with nodes in the
        subset.
    remove_incoming_edges(y, new)
        Remove all incoming edges of all nodes in y. If new, return a new
        graph.
    remove_outgoing_edges(y, new)
        Remove all outgoing edges of all nodes in y.
    """

    def __init__(self, causation, observed, graph=None):
        self.causation = causation
        self.observed_var = set(observed)
        self.unobserved_var = set(causation.keys()) - self.observed_var

        if graph is None:
            edges = []
            for k, v in causation.items():
                for para in v:
                    edges.append((para, k))
            self.dag = nx.DiGraph()
            self.dag.add_edges_from(edges)
        else:
            self.dag = graph

    @property
    def prob(self):
        return prob.Prob(variables=self.observed_var)

    @property
    def is_dag(self):
        # TODO: determin if the graph is a DAG, try tr(e^{W\circledot W}-d)=0
        return nx.is_directed_acyclic_graph(self.dag)

    def add_nodes(self, nodes, new=False):
        if not new:
            self.dag.add_nodes_from(nodes)
            for node in nodes:
                self.causation[node] = []
                self.observed_var.add(node)
        else:
            pass

    def add_edges_from(self, edge_list, new=False):
        """Add edges to the causal graph.

        Parameters
        ----------
        edge_list : list
            every element of the list contains two elements, the first for
            the parent
        new : bool
            return a new graph if set as True
        """
        if not new:
            self.dag.add_edges_from(edge_list)
            for edge in edge_list:
                self.causation[edge[1]].append(edge[0])
        else:
            pass

    def add_edge(self, i, j):
        self.dag.add_edge(i, j)
        self.causation[j].append(i)

    def remove_nodes(self, nodes, new=False):
        if not new:
            for node in nodes:
                self.observed_var.remove(node)
                for k, v in self.causation:
                    if k == node:
                        del self.causation[node]
                        continue
                    v.remove(node)
            self.dag.remove_nodes_from(nodes)
        else:
            new_observed_var = set(self.observed_var)
            new_causation = dict(self.causation)
            new_dag = self.dag.copy()
            new_dag.remove_nodes_from(nodes)

            for node in nodes:
                new_observed_var.remove(node)
                for k, v in new_causation:
                    if k == node:
                        del new_causation[node]
                        continue
                    v.remove(node)
            return CausalGraph(new_causation, new_observed_var, new_dag)

    def remove_edges_from(self, edge_list, new=False):
        if not new:
            self.dag.remove_edges_from(edge_list)
            for edge in edge_list:
                self.causation[edge[1]].remove(edge[0])
        else:
            new_dag = self.dag.copy()
            new_causation = self.causation
            new_oberser = set(self.observed_var)
            new_dag.remove_edges_from(edge_list)
            for edge in edge_list:
                new_causation[edge[1]].remove(edge[0])
            return CausalGraph(new_causation, new_oberser, new_dag)

    def remove_edge(self, i, j, new=False):
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
        an = set()
        for node in x:
            an.add(node)
            an.update(nx.ancestors(self.dag, node))
        return an

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
        return nx.topological_sort(self.dag)

    def build_sub_graph(self, subset):
        """Construct the subgraph with the nodes in subset"""
        nodes = set(self.causation.keys()).difference(subset)
        return self.remove_nodes(nodes, new=True)

    def remove_incoming_edges(self, x, new=False):
        """remove incoming edges of all nodes in x.

        Parameters
        ----------
        x : set
        new : bool
            return a new graph if set as Ture

        Returns
        ----------
        modified_graph : CausalGraph
            subgraph of the graph without all incoming edges of nodes in x
        """
        return self.remove_edges_from(
            list(self.dag.in_edges(x)), new
        )

    def remove_outgoing_edges(self, x, new=True):
        """remove outcoming edges of all nodes in x.

        Parameters
        ----------
        x : set
        new : bool

        Returns
        ----------
        modified_graph : CausalGraph
            subgraph of the graph without all outcoming edges of nodes in x
        """
        return self.remove_edges_from(
            list(self.dag.out_edges(x)), new
        )
