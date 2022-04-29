import networkx as nx
import numpy as np

from copy import deepcopy
from collections import defaultdict
from . import prob


class CausalGraph:
    """
    Class for building a causal graphical model.

    Attributes
    ----------
    causation : dict
        Data structure of the causal graph where values are parents of the
        corresponding keys.
    dag : nx.MultiDiGraph
        Graph represented by the networkx package.
    prob
    latent_confounding_arcs
    is_dag
    c_components
    observed_dag
    topo_order
    explicit_unob_var_dag

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

    def __init__(self, causation, dag=None, latent_confounding_arcs=None):
        """
        Parameters
        ----------
        causation : dict
            Data structure of the causation
        graph : nx.MultiGraph, optional
            The causal graph. Defaults to None.
        latent_confounding_arcs : set or list, optional
            Unobserved bidirected edges. Defaults to None. Each element is a
            tuple containing 2 elements.
        """
        # TODO: replace list or tuple with generator to save memory

        self.causation = defaultdict(list, causation)
        self.dag = self.observed_dag.copy() if dag is None else dag

        # add unobserved bidirected confounding arcs to the graph, with the
        # letter 'n' representing that the edge is unobserved
        if latent_confounding_arcs is not None:
            for edge in latent_confounding_arcs:
                self.dag.add_edges_from(
                    [(edge[0], edge[1], 'n'), (edge[1], edge[0], 'n')]
                )

    @property
    def prob(self):
        """The encoded probability distribution.

        Returns
        ----------
        Prob
        """
        return prob.Prob(variables=set(self.causation.keys()))

    @property
    def latent_confounding_arcs(self):
        """Return the latent confounding arcs encoded in the graph.

        Returns
        ----------
        list

        """
        W = nx.to_numpy_matrix(self.dag)
        a, w_t = np.where(W >= 1), W.T.A
        arcs, nodes = [], list(self.dag.nodes)
        for row, col in zip(a[0], a[1]):
            if w_t[row][col] >= 1 and (nodes[col], nodes[row]) not in arcs:
                arcs.append((nodes[row], nodes[col]))
        return arcs

    @property
    def is_dag(self):
        """Verify whether the constructed graph is a DAG.
        """
        # TODO: determin if the graph is a DAG, try tr(e^{W\circledot W}-d)=0
        return nx.is_directed_acyclic_graph(self.observed_dag)

    def to_adj_matrix(self):
        """Return the adjacency matrix.
        """
        W = nx.to_numpy_matrix(self.dag)
        return W

    def to_adj_list(self):
        """Return the adjacency list."""
        pass

    def is_d_separated(self, x, y, test_set):
        """Check if test_set d-separates x and y.

        Parameters
        ----------
        x : set of str
        y : set of str
        test_set : set of str

        Returns
        ----------
        Bool
            If test_set d-separates x and y, return True else return False.
        """
        return nx.d_separated(self.explicit_unob_var_dag, x, y, test_set)

    @property
    def c_components(self):
        """Return the C-component set of the graph.

        Returns
        ----------
        set of str
            The C-component set of the graph
        """
        bi_directed_graph = nx.Graph()
        bi_directed_graph.add_nodes_from(self.dag.nodes)
        bi_directed_graph.add_edges_from(self.latent_confounding_arcs)
        return nx.connected_components(bi_directed_graph)

    def ancestors(self, x):
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
        for node in x:
            an.add(node)
            try:
                an.update(nx.ancestors(self.observed_dag, node))
            except Exception:
                pass
        return an

    @property
    def observed_dag(self):
        """Return the observed part of the graph, including observed nodes and
        edges between them.

        Returns
        ----------
        nx.MultiDiGraph
            The observed part of the graph
        """
        edges = []
        for k, v in self.causation.items():
            for para in v:
                edges.append((para, k, 0))
        ob_dag = nx.MultiDiGraph()
        ob_dag.add_edges_from(edges)
        return ob_dag

    @property
    def explicit_unob_var_dag(self):
        """Build a new dag where all unobserved confounding arcs are replaced
        by explicit unobserved variables

        Returns
        ----------
        nx.MultiDiGraph
        """
        new_dag = self.observed_dag
        for i, (node1, node2) in enumerate(self.latent_confounding_arcs):
            new_dag.add_edges_from(
                [(f'U{i}', node1, 'n'), (f'U{i}', node2, 'n')]
            )
        return new_dag

    @property
    def topo_order(self):
        """Retrun the topological order of the nodes in the observed graph

        Returns
        ----------
        generator
            Nodes in the topological order
        """
        return nx.topological_sort(self.observed_dag)

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
        ori_nodes = self.dag.nodes

        if not new:
            self.dag.add_nodes_from(nodes)
            for node in nodes:
                if node not in ori_nodes:
                    self.causation[node] = []
        else:
            new_dag = deepcopy(self.dag)
            new_causation = deepcopy(self.causation)
            new_dag.add_nodes_from(nodes)
            for node in nodes:
                if node not in ori_nodes:
                    new_causation[node] = []
            return CausalGraph(new_causation, dag=new_dag)

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
            new_dag = deepcopy(self.dag)
            new_causation = deepcopy(self.causation)
            if observed:
                new_dag.add_edges_from(edge_list)
                for edge in edge_list:
                    new_causation[edge[1]].append(edge[0])
            else:
                for edge in edge_list:
                    new_dag.add_edges_from(
                        [(edge[0], edge[1], 'n'), (edge[1], edge[0], 'n')]
                    )
            return CausalGraph(new_causation, dag=new_dag)

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
            Add an unobserved latent confounding arc if False.
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
                for k in list(self.causation.keys()):
                    if k == node:
                        del self.causation[node]
                        continue
                    try:
                        self.causation[k].remove(node)
                    except Exception:
                        pass
            self.dag.remove_nodes_from(nodes)
        else:
            new_causation = deepcopy(self.causation)
            new_dag = deepcopy(self.dag)
            new_dag.remove_nodes_from(nodes)

            for node in nodes:
                for k in list(new_causation.keys()):
                    if k == node:
                        del new_causation[node]
                        continue
                    try:
                        new_causation[k].remove(node)
                    except Exception:
                        pass
            return CausalGraph(new_causation, dag=new_dag)

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
            self.dag.remove_edge(edge[0], edge[1], 0)
            try:
                self.causation[edge[1]].remove(edge[0])
            except Exception:
                pass
        else:
            self.dag.remove_edges_from(
                [(edge[0], edge[1], 'n'), (edge[1], edge[0], 'n')]
            )

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
            Return a new CausalGraph if new.
        """
        if not new:
            if observed:
                for edge in edge_list:
                    self.dag.remove_edge(edge[0], edge[1], 0)
                    try:
                        self.causation[edge[1]].remove(edge[0])
                    except Exception:
                        pass
            else:
                for edge in edge_list:
                    self.dag.remove_edges_from(
                        [(edge[0], edge[1], 'n'), (edge[1], edge[0], 'n')]
                    )
        else:
            new_dag = deepcopy(self.dag)
            new_causation = deepcopy(self.causation)
            if observed:
                for edge in edge_list:
                    new_dag.remove_edge(edge[0], edge[1], 0)
                    try:
                        new_causation[edge[1]].remove(edge[0])
                    except Exception:
                        pass
            else:
                for edge in edge_list:
                    new_dag.remove_edges_from(
                        [(edge[0], edge[1], 'n'), (edge[1], edge[0], 'n')]
                    )
            return CausalGraph(new_causation, new_dag)

    def build_sub_graph(self, subset):
        """Return a new CausalGraph as the subgraph of the graph with nodes in the
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
        """Remove incoming edges of all nodes of x. If new, do this in the new
        CausalGraph.

        Parameters
        ----------
        x : set or list
        new : bool
            Return a new graph if set as Ture.

        Returns
        ----------
        CausalGraph
            If new, return a subgraph of the graph without all incoming edges
            of nodes in x
        """
        edges = self.dag.in_edges(x, keys=True)
        o_edges, u_edges = [], []

        for edge in edges:
            if edge[2] == 'n':
                u_edges.append(edge)
            else:
                o_edges.append(edge)

        if new:
            return self.remove_edges_from(o_edges, new).remove_edges_from(
                u_edges, new, observed=False
            )
        else:
            self.remove_edges_from(o_edges, new)
            self.remove_edges_from(u_edges, new, observed=False)

    def remove_outgoing_edges(self, x, new=False):
        """Remove outcoming edges of all nodes in x.

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
        removing_edges = [
            edge for edge in self.dag.out_edges(x, keys=True) if edge[2] == 0
        ]
        return self.remove_edges_from(removing_edges, new, observed=True)
