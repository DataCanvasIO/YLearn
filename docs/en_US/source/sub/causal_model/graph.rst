.. _causal_graph:

************
Causal Graph
************

This is a class for representing DAGs of causal structures.

Generally, for a 
set of variables :math:`V`, a variable :math:`V_i` is said to be a cause of a variable :math:`V_j`
if :math:`V_j` can change in response to changes in :math:`V_i`. In a DAG for causal
structures, every parent is a direct causes of all its children. And we refer to these DAGs for
causal structures as causal graphs. For the terminologies of graph, one can see, for example,
Chapter 1.2 in [Pearl]_.

There are five basic structures composed of two or three nodes for building causal graphs. Besides the structures, 
there are flows of association and causation in causal graphs in the probability language. Any two nodes 
:math:`X` and :math:`Y` connected by the flow of association implies that they are statistically dependent, i.e.,
:math:`P(X, Y) \neq P(X)P(Y)`. Let :math:`X, Y` and :math:`W` be three distinct nodes, then the five basics
structures include:

1. *chains*:

.. math::

    X \rightarrow W \rightarrow Y,

:math:`X` and :math:`Y` are statistically dependent;

2. *forks*:

.. math::

    X \leftarrow W \rightarrow Y,

:math:`X` and :math:`Y` are statistically dependent;

3. *colliders*:

.. math::

    X \rightarrow W \leftarrow Y,

:math:`X` and :math:`Y` are statistically independent;

4. *two unconnected nodes*:

.. math:: 

    X \quad Y,

:math:`X` and :math:`Y` are statistically independent;

5. *two connected nodes*:

.. math::

    X \rightarrow Y,

:math:`X` and :math:`Y` are statistically dependent.

In YLearn, one can use the :class:`CausalGraph` to represent causal structures by first giving a python dict where
each key in this dict is a child of all elements in the corresponding dict value, which usually should be a list 
of str.

.. topic:: Examples

    .. figure:: graph_expun.png
        :scale: 40 %

        Causal structures where all green nodes are unobserved (a variable is unobserved if it is not present
        in the dataset but one believes that it will have causal relationships with other Variables).

    .. figure:: graph_un_arc.png
        
        Causal structures where all unobserved variables are removed and their related causations are replaced by
        the confounding arcs (black doted lines with two arrows).
    
    We can represent this causal structure with YLearn as follows:

    .. code-block:: python
        
        from ylearn.causal_model.graph import CausalGraph
        causation = {
            'X': ['Z2'],
            'Z1': ['X', 'Z2'],
            'Y': ['Z1', 'Z3'],
            'Z3': ['Z2'],
            'Z2': [], 
        }
        arcs = [('X', 'Z2'), ('X', 'Z3'), ('X', 'Y'), ('Z2', 'Y')]

        cg = CausalGraph(causation=causation, latent_confounding_arcs=arcs)

        list(cg.c_components)
    
    >>> [{'X', 'Y', 'Z2', 'Z3'}, {'Z1'}]

Class Structures
================

.. py:class:: ylearn.causal_model.graph.CausalGraph(causation, dag=None, latent_confounding_arcs=None)

    :param dict causation: Descriptions of the causal structures where values are parents of the
            corresponding keys.
    :param networkx.MultiGraph, optional, default=None dag: A known graph structure.
            If provided, dag must represent the causal structures stored in causation.
    :param set or list of tuple of two str, optional, default=None, latent_confounding_arcs: Two elements 
            in the tuple are names of nodes in the graph where there
            exists an latent confounding arcs between them. Semi-Markovian graphs
            with unobserved confounders can be converted to a graph without
            unobserved variables, where one can add bi-directed latent confounding
            arcs to represent these relations. For example, the causal graph X <- U -> Y,
            where U is an unobserved confounder of X and Y, can be converted
            equivalently to X <-->Y where <--> is a latent confounding arc.

    .. py:method:: ancestors(x)
        
        Return the ancestors of all nodes in x.
        
        :param set of str x: A set of nodes in the graph.

        :returns: Ancestors of nodes in x in the graph.
        :rtype: set of str

    .. py:method:: descendents(x)
        
        Return the descendents of all nodes in x.
        
        :param set of str x: A set of nodes in the graph.

        :returns: Descendents of nodes in x in the graph.
        :rtype: set of str

    .. py:method:: parents(x, only_observed=True)
        
        Return the direct parents of the node x in the graph.        
        
        :param str x: Name of the node x.
        :param bool, default=True only_observed: If True, then only find the observed parents in the causal graph,
                otherwise also include the unobserved variables, by default True

        :returns: Parents of the node x in the graph
        :rtype: list

    .. py:method:: add_nodes(nodes, new=False)
        
        If not new, add all nodes in the nodes to the current
        CausalGraph, else create a new graph and add nodes.
        
        :param set or list x: Nodes waited to be added to the current causal graph.
        :param bool, default=False new: If new create and return a new graph. Defaults to False.
        
        :returns: Modified causal graph
        :rtype: instance of CausalGraph

    .. py:method:: add_edges_from(edge_list, new=False, observed=True)
        
        Add edges to the causal graph.
        
        :param list edge_list: Every element of the list contains two elements, the first for
                the parent
        :param bool, default=False new: If new create and return a new graph. Defaults to False.
        :param bool, default=True observed: Add unobserved bidirected confounding arcs if not observed.
        
        :returns: Modified causal graph
        :rtype: instance of CausalGraph

    .. py:method:: add_edge(edge_list, s, t, observed=True)
        
        Add edges to the causal graph.
        
        :param str s: Source of the edge.
        :param str t: Target of the edge.
        :param bool, default=True observed: Add unobserved bidirected confounding arcs if not observed.

    .. py:method:: remove_nodes(nodes, new=True)
        
        Remove all nodes of nodes in the graph.

        :param set or list nodes: Nodes waited to be removed.
        :param bool, default=True new: If True, create a new graph, remove nodes in that graph and return
                it. Defaults to False.

        :returns: Modified causal graph
        :rtype: instance of CausalGraph

    .. py:method:: remove_edge(edge, observed=True)
        
        Remove the edge in the CausalGraph. If not observed, remove the unobserved
        latent confounding arcs.

        :param tuple edge: 2 elements denote the start and end of the edge, respectively.
        :param bool, default=True observed: If not observed, remove the unobserved latent confounding arcs.

    .. py:method:: remove_edges_from(edge_list, new=False, observed=True)
        
        Remove all edges in the edge_list in the graph.

        :param list edge_list: list of edges to be removed.
        :param bool, default=False new: If new, create a new CausalGraph and remove edges.
        :param bool, default=True observed: Remove unobserved latent confounding arcs if not observed.

        :returns: Modified causal graph
        :rtype: instance of CausalGraph

    .. py:method:: build_sub_graph(subset)
        
        Return a new CausalGraph as the subgraph of the graph with nodes in the
        subset.

        :param set subset: The set of the subgraph.

        :returns: Modified causal graph
        :rtype: instance of CausalGraph

    .. py:method:: remove_incoming_edges(x, new=False)
        
        Remove incoming edges of all nodes of x. If new, do this in the new
        CausalGraph.

        :param set or list x:
        :param bool, default=False, new: Return a new graph if set as Ture.

        :returns: Modified causal graph
        :rtype: instance of CausalGraph

    .. py:method:: remove_outgoing_edges(x, new=False)
        
        Remove outgoing edges of all nodes of x. If new, do this in the new
        CausalGraph.

        :param set or list x:
        :param bool, default=False, new: Return a new graph if set as Ture.

        :returns: Modified causal graph
        :rtype: instance of CausalGraph

    .. py:property:: c_components
        
        The C-components set of the graph.
        
        :returns: The C-components set of the graph.
        :rtype: set of str

    .. py:property:: observed_dag
        
        Return the observed part of the graph, including observed nodes and
        edges between them.
        
        :returns: The observed part of the graph
        :rtype: networkx.MultiGraph

    .. py:property:: explicit_unob_var_dag
        
        Build a new dag where all unobserved confounding arcs are replaced
        by explicit unobserved variables.
        
        :returns: Dag with explicit unobserved nodes
        :rtype: networkx.MultiGraph   
    
    .. py:property:: topo_order

        Return the topological order of the nodes in the observed graph.
        
        :returns: Nodes in the topological order
        :rtype: generator          