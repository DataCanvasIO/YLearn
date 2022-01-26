import causal_model
import networkx as nx
import numpy as np
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

    def __init__(self, causation, graph=None, latent_confounding_arcs=None):
        self.causation = causation
        # self.observed_var = set(observed)
        # self.unobserved_var = set(causation.keys()) - self.observed_var
        # TODO: update the usage of DiGraph with MultiDiGraph, consider all
        # usages, or simply add new unobserved nodes
        # TODO: now only consider confounding arc for unobserved variables,
        # what about unobserved chain?
        # TODO: replace list or tuple with generator to save memory
        if graph is None:
            edges = []
            for k, v in causation.items():
                for para in v:
                    edges.append((para, k))
            self.dag = nx.MultiDiGraph()
            self.dag.add_edges_from(edges)
        else:
            self.dag = graph

        self.init_confounding_arcs = latent_confounding_arcs
        if latent_confounding_arcs is not None:
            for edge in latent_confounding_arcs:
                self.add_edges_from(
                    [(edge[0], edge[1], 'n'), (edge[1], edge[0], 'n')]
                )

    @property
    def prob(self):
        return prob.Prob(variables=self.causation.keys())

    @property
    def latent_confounding_arcs(self):
        W = nx.to_numpy_matrix(self.dag)
        a, b = np.where(W >= 1), np.where(W.T >= 1)
        arcs, nodes = [], list(self.dag.nodes)
        for i, (j, k) in enumerate(zip(a[0] == b[0], a[1] == b[1])):
            if j and k:
                arcs.append(
                    (
                        nodes[a[0][i]], nodes[a[1][i]]
                    )
                )
        return arcs

    @property
    def is_dag(self):
        # TODO: determin if the graph is a DAG, try tr(e^{W\circledot W}-d)=0
        return nx.is_directed_acyclic_graph(self.observed_part)

    def add_nodes(self, nodes, new=False):
        if not new:
            self.dag.add_nodes_from(nodes)
            for node in nodes:
                self.causation[node] = []
        else:
            new_dag = self.dag.copy()
            new_causation = dict(self.causation)
            new_dag.add_nodes_from(nodes)
            for node in nodes:
                new_causation[node] = []
            return CausalGraph(new_causation, graph=new_dag)

    def add_edges_from(self, edge_list, new=False, observed=True):
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
            if observed:
                for edge in edge_list:
                    self.causation[edge[1]].append(edge[0])
                    self.dag.add_edge(edge[0], edge[1], 0)
            else:
                for edge in edge_list:
                    self.dag.add_edges_from(
                        [(edge[0], edge[1], 'n'), (edge[1], edge[0], 'n')]
                    )
        else:
            new_dag = self.dag.copy()
            new_causation = dict(self.causation)
            if observed:
                new_dag.add_edges_from(edge_list)
                for edge in edge_list:
                    new_causation[edge[1]].append(edge[0])
            else:
                for edge in edge_list:
                    new_dag.add_edges_from(
                        [(edge[0], edge[1], 'n'), (edge[1], edge[0], 'n')]
                    )
            return CausalGraph(new_causation, graph=new_dag)

    def add_edge(self, i, j, observed=True):
        if observed:
            self.dag.add_edge(i, j, 0)
            self.causation[j].append(i)
        else:
            self.dag.add_edge(i, j, 'n')

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

    def remove_edges_from(self, edge_list, new=False, observed=True):
        if not new:
            if observed:
                for edge in edge_list:
                    self.dag.remove_edge(edge[0], edge[1], 0)
                    self.causation[edge[1]].remove(edge[0])
            else:
                for edge in edge_list:
                    self.dag.remove_edges_from(
                        [(edge[0], edge[1], 'n'), (edge[1], edge[0], 'n')]
                    )
        else:
            new_dag = self.dag.copy()
            new_causation = self.causation
            if observed:
                for edge in edge_list:
                    new_dag.remove_edge(edge[0], edge[1], 0)
                    new_causation[edge[1]].remove(edge[0])
            else:
                for edge in edge_list:
                    new_dag.remove_edges_from(
                        [(edge[0], edge[1], 'n'), (edge[1], edge[0], 'n')]
                    )
            return CausalGraph(new_causation, new_dag)

    def remove_edge(self, edge, new=False, observed=True):
        if not observed:
            self.dag.remove_edge(edge[0], edge[1], 'n')
        else:
            self.dag.remove_edges(edge[0], edge[1])
            self.causation[edge[1]].remove(edge[0])

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
        bi_directed_graph = nx.Graph()
        bi_directed_graph.add_edges_from(self.latent_confounding_arcs)
        return nx.connected_components(bi_directed_graph)

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
            an.update(nx.ancestors(self.observed_part, node))
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
        return nx.topological_sort(self.observed_part)

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
        # TODO: consider wether remove the bi-directed unobserved edges
        edges = list(self.dag.in_edges(x, keys=True))
        u_edges = []

        for i, edge in enumerate(edges):
            if edge[2] == 'n':
                u_edges.append(edges.pop(i))

        if new:
            return self.remove_edges_from(edges, new).remove_edges_from(
                u_edges, new, observed=False
            )
        else:
            self.remove_edges_from(edges, new)
            self.remove_edges_from(u_edges, new, observed=False)

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
        edges = list(self.dag.out_edges(x, keys=True))
        for i, edge in enumerate(edges):
            if edge[2] == 'n':
                edges.pop(i)
        return self.remove_edges_from(edges, new, observed=True)
