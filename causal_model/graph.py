import causal_model
import networkx as nx
import numpy as np

from causal_model import prob
from estimator_model.estimation_learner.meta_learner import SLearner


class CausalGraph:
    """Causal Graph.

    Attributes
    ----------
    causation : dic
        Data structure of the causal graph where values are parents of the
        corresponding keys.
    observed_var : list
    unobserved_var : list
    dag : nx.DiGraph
        Graph represented by the networkx package.
    prob
    latent_confounding_arcs
    is_dag
    c_components
    observed_graph
    topo_order

    Methods
    ----------
    to_adj_matrix()
        Return the numpy matrix of the adjecency matrix.
    to_adj_list()
        Return the numpy array of the adjecency matrix.
    ancestors(y)
        Return ancestors of y.
    add_nodes(nodes, new=False)
        If not new, add all nodes in the nodes to the current
        CausalGraph, else create a new graph and add nodes.
    add_edges_from(edge_list, new=False, observed=True)
        Add all edges in the edge_list to the CausalGraph.
    add_edge(i, j, observed=True)
        Add an edge between nodes i and j to the CausalGraph. Add an unobserved
        confounding arc if not observed.
    remove_nodes(nodes, new=False)
        Remove all nodes in the graph. If new, do this in a new CausalGraph.
    remove_edge(i, j, observed=True)
        Remove the edge in the CausalGraph. If observed, remove the unobserved
        latent confounding arcs.
    remove_edges_from(edge_list, new=False, observed=True)
        Remove all edges in the edge_list in the CausalGraph.
    build_sub_graph(subset)
        Return a new CausalGraph as the subgraph of self with nodes in the
        subset.
    remove_incoming_edges(y, new=False)
        Remove all incoming edges of all nodes in y. If new, return a new
        CausalGraph.
    remove_outgoing_edges(y, new=False)
        Remove all outgoing edges of all nodes in y. If new, return a new
        CausalGraph.
    """

    def __init__(self, causation, graph=None, latent_confounding_arcs=None):
        """
        Parameters
        ----------
        causation : dict
            data structure of the causation
        graph : nx.MultiGraph, optional
            the causal graph. Defaults to None.
        latent_confounding_arcs : set or list, optional
            unobserved bidirected edges. Defaults to None.
        """
        # TODO: update the usage of DiGraph with MultiDiGraph, consider all
        # usages, or simply add new unobserved nodes
        # TODO: now only consider confounding arc for unobserved variables,
        # what about unobserved chain?
        # TODO: replace list or tuple with generator to save memory

        self.causation = causation

        if graph is None:
            self.dag = self.observed_graph.copy()
        else:
            self.dag = graph

        # add unobserved bidirected confounding arcs to the graph, the letter
        # 'n' representing that the edge is unobserved
        if latent_confounding_arcs is not None:
            for edge in latent_confounding_arcs:
                self.dag.add_edges_from(
                    [(edge[0], edge[1], 'n'), (edge[1], edge[0], 'n')]
                )

    @property
    def prob(self):
        """
        The encoded probability distribution.

        Returns
        ----------
        Prob
        """
        return prob.Prob(variables=self.causation.keys())

    @property
    def latent_confounding_arcs(self):
        """
        Return the latent confounding arcs encoded in the graph.

        Returns
        ----------
        arcs : list

        """
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
        """Determine whether the constructed graph is a DAG.
        """
        # TODO: determin if the graph is a DAG, try tr(e^{W\circledot W}-d)=0
        return nx.is_directed_acyclic_graph(self.observed_graph)

    def to_adj_matrix(self):
        """Return the adjacency matrix.
        """
        W = nx.to_numpy_matrix(self.dag)
        return W

    def to_adj_list(self):
        """Return the adjacency list."""
        pass

    @property
    def c_components(self):
        """
        Return the C-component set of the graph.

        Returns
        ----------
        c : set
            the C-component set of graph
        """
        bi_directed_graph = nx.Graph()
        bi_directed_graph.add_edges_from(self.latent_confounding_arcs)
        return nx.connected_components(bi_directed_graph)

    def ancestors(self, x):
        """
        Return the ancestors of all nodes in x.

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
            an.update(nx.ancestors(self.observed_graph, node))
        return an

    @property
    def observed_graph(self):
        """
        Return the observed part of the graph, including observed nodes and
        edges between them.

        Returns
        ----------
        ob_graph : CausalGraph
            the observed part of the graph
        """
        edges = []
        for k, v in self.causation.items():
            for para in v:
                edges.append((para, k))
        ob_graph = nx.MultiDiGraph()
        ob_graph.add_edges_from(edges)
        return ob_graph

    @property
    def topo_order(self):
        """
        Retrun the topological order of the nodes in the observed graph

        Returns
        ----------
        topological_order : generator
            nodes in the topological order
        """
        return nx.topological_sort(self.observed_graph)

    def add_nodes(self, nodes, new=False):
        """
        If not new, add all nodes in the nodes to the current
        CausalGraph, else create a new graph and add nodes.

        Parameters
        ----------
        nodes : set or list
        new : bool, optional
            If new create and return a new graph. Defaults to False.

        Returns
        ----------
        CausalGraph
        """
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
        """
        Add edges to the causal graph.

        Parameters
        ----------
        edge_list : list
            Every element of the list contains two elements, the first for
            the parent
        new : bool
            Return a new graph if set as True
        observed : bool
            Add unobserved bidirected confounding arcs if not observed.
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

    def add_edge(self, s, t, observed=True):
        """
        Add an edge between nodes i and j. Add an unobserved latent confounding
        arc if not observed.

        Parameters
        ----------
        s : str
            Source of the edge.
        t : str
            Target of the edge.
        observed : bool
            Add an unobserved latent confounding arc if True.
        """
        if observed:
            self.dag.add_edge(s, t, 0)
            self.causation[t].append(s)
        else:
            self.dag.add_edge(s, t, 'n')

    def remove_nodes(self, nodes, new=False):
        """
        Remove all nodes in the graph.

        Parameters
        ----------
        nodes : set or list
        new : bool, optional
            If True, create a new graph, remove nodes in that graph and return
            it. Defaults to False.

        Returns
        ---------
        CausalGraph
            Return a CausalGraph if new.
        """
        if not new:
            for node in nodes:
                # self.observed_var.remove(node)
                for k, v in self.causation:
                    if k == node:
                        del self.causation[node]
                        continue
                    v.remove(node)
            self.dag.remove_nodes_from(nodes)
        else:
            # new_observed_var = set(self.observed_var)
            new_causation = dict(self.causation)
            new_dag = self.dag.copy()
            new_dag.remove_nodes_from(nodes)

            for node in nodes:
                # new_observed_var.remove(node)
                for k, v in new_causation:
                    if k == node:
                        del new_causation[node]
                        continue
                    v.remove(node)
            return CausalGraph(new_causation, graph=new_dag)

    def remove_edge(self, edge, observed=True):
        """
        Remove the edge in the CausalGraph. If observed, remove the unobserved
        latent confounding arcs.

        Parameters
        ----------
        edge : tuple
            2 elements.
        observed : bool
            If not observed, remove the unobserved latent confounding arcs.
        """
        if observed:
            self.dag.remove_edges(edge[0], edge[1], 0)
            self.causation[edge[1]].remove(edge[0])
        else:
            self.dag.remove_edge(edge[0], edge[1], 'n')

    def remove_edges_from(self, edge_list, new=False, observed=True):
        """
        Remove all edges in the edge_list in the graph.

        Parameters
        ----------
        edge_list : list
        new : bool, optional
            If new, creat a new CausalGraph and remove edges.
        observed : bool, optional
            Remove unobserved latent confounding arcs if not observed.

        Returns
        ----------
        CausalGraph
            If not observed, return a new CausalGraph.
        """
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

    def build_sub_graph(self, subset):
        """
        Return a new CausalGraph as the subgraph of graph with nodes in the
        subset.

        Parameters
        ----------
        subset : set

        Returns
        ----------
        CausalGraph
        """
        nodes = set(self.causation.keys()).difference(subset)
        return self.remove_nodes(nodes, new=True)

    def remove_incoming_edges(self, x, new=False):
        """
        Remove incoming edges of all nodes of x. If new, do this in the new
        CausalGraph.

        Parameters
        ----------
        x : set
        new : bool
            Return a new graph if set as Ture.

        Returns
        ----------
        CausalGraph
            If new, return a subgraph of the graph without all incoming edges
            of nodes in x
        """
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

    def remove_outgoing_edges(self, x, new=False):
        """
        Remove outcoming edges of all nodes in x.

        Parameters
        ----------
        x : set
        new : bool

        Returns
        ----------
        CausalGraph
            If new, return a subgraph of the graph without all outcoming edges
            of nodes in x.
        """
        edges = list(self.dag.out_edges(x, keys=True))
        for i, edge in enumerate(edges):
            if edge[2] == 'n':
                edges.pop(i)
        return self.remove_edges_from(edges, new, observed=True)
