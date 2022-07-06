.. _causal_graph:

************
因果图
************

这是一个表示因果结构的DAGs的类。

通常来说，对于一组变量 :math:`V` ，如果 :math:`V_j` 能够对应于 :math:`V_i` 的变化而变化，那么变量 :math:`V_i` 被称为变量 :math:`V_j` 的因。在
一个因果结构的DAG中，每个父节点都是它所有的孩子的直接的因。我们把这些因果结构的DAGs称为因果图。对于图的术语，举个例子，可以参考 [Pearl]_ 的Chapter 1.2。

有五个基本的由两到三个节点组成的结构用于构建因果图。除了这些结构，还有用概率的语言描述的在因果图中的关联和因果关系的流。任意两个节点 :math:`X`
和 :math:`Y` ，如果被关联流连接在一起，则表示它们是统计相关的。等价于 :math:`P(X, Y) \neq P(X)P(Y)` 。令 :math:`X, Y` 和 :math:`W`
为三个不同的节点，那么五个基本的结构包括：

1. *链*:

.. math::

    X \rightarrow W \rightarrow Y,

:math:`X` 和 :math:`Y` 是统计相关的;

2. *叉*:

.. math::

    X \leftarrow W \rightarrow Y,

:math:`X` 和 :math:`Y` 是统计相关的;

3. *对撞*:

.. math::

    X \rightarrow W \leftarrow Y,

:math:`X` 和 :math:`Y` 是统计独立的;

4. *两个不相连的节点*:

.. math:: 

    X \quad Y,

:math:`X` 和 :math:`Y` 是统计独立的；

5. *两个相连的节点*:

.. math::

    X \rightarrow Y,

:math:`X` 和 :math:`Y` 是统计相关的。

在YLearn中，使用 :class:`CausalGraph` 来表示因果结构，首先给一个python的字典，其中每个键都是它对应的通常由字符串的列表表示的值的每个元素的子节点。

.. topic:: 例子

    .. figure:: graph_expun.png
        :scale: 40 %

        因果结构，其中所有的绿色节点都是未观察到的（一个变量是未观察到的如果它没有显示在数据集中但是可以相信它与其他变量有因果关系）。

    .. figure:: graph_un_arc.png

        因果结构，其中所有的未观察到的变量都被移除了，它们相关的因果关系都被混淆的弧线（有两个箭头的黑色虚线）取代了。
    
    我们可以像下面那样用YLearn表示这个因果结构：

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

类结构
================

.. py:class:: ylearn.causal_model.graph.CausalGraph(causation, dag=None, latent_confounding_arcs=None)

    :param dict causation: 描述因果结构，其中值是对应的键的父节点。
    :param networkx.MultiGraph, optional, default=None dag: 一个已知的图结构。如果提供了，DAG必须表示存储在因果关系中的因果结构。
    :param set or list of tuple of two str, optional, default=None, latent_confounding_arcs: 元组中的两个元素是图中节点的名字
            其中它们之间存在潜在的混淆弧线。有未观测的混淆因素的半马尔可夫图能够被转化为一个没有未观测到变量的图，其中可以添加双向潜在混淆弧线
            表示这些关系。比如，在因果图X <- U -> Y，其中U是一个未观测到的X和Y的混淆因素，可以被等价的转变为X <--> Y，其中<-->是一个潜在的混淆弧线。

    .. py:method:: ancestors(x)
        
        返回x中所有节点的祖先。
        
        :param set of str x: 图中的一组节点。

        :returns: 图中x中节点的祖先。
        :rtype: 一组str

    .. py:method:: descendents(x)
        
        返回x中所有节点的后代。
        
        :param set of str x: 图中的一组节点。

        :returns: 图中x中节点的后代。
        :rtype: 一组str

    .. py:method:: parents(x, only_observed=True)
        
        返回图中x节点的直接父节点。
        
        :param str x: x节点的名字.
        :param bool, default=True only_observed: 如果为True，那么仅找到观测到的在因果图中的父节点，否则，也包含未观测到的变量，默认是True。

        :returns: 图中x节点的父节点
        :rtype: 列表

    .. py:method:: add_nodes(nodes, new=False)
        
        如果new是False，则把nodes中所有的节点加入到现在的CausalGraph，否则创建一个新图并加入节点。
        
        :param set or list x: 等待被加入到现在的因果图的节点
        :param bool, default=False new: 如果是新创建，则返回一个新的图。默认是False。
        
        :returns: 修改的因果图
        :rtype: CausalGraph的实例

    .. py:method:: add_edges_from(edge_list, new=False, observed=True)
        
        在因果图中加入边。
        
        :param list edge_list: 列表中的每个元素包含两个元素，第一个元素是父节点
        :param bool, default=False new: 如果是新创建，则返回一个新的图。默认是False。
        :param bool, default=True observed: 如果未观测到，添加未观测到的双向混淆弧线。
        
        :returns: 修改的因果图
        :rtype: CausalGraph的实例

    .. py:method:: add_edge(edge_list, s, t, observed=True)
        
        在因果图中加入边。
        
        :param str s: 边的源。
        :param str t: 边的目的。
        :param bool, default=True observed: 如果未观测到，添加未观测到的双向混淆弧线。

    .. py:method:: remove_nodes(nodes, new=True)
        
        把nodes所有的节点从图中移除。

        :param set or list nodes: 等待移除的节点。
        :param bool, default=True new: 如果为True，创建一个新图，移除图中的节点并返回。默认是False。

        :returns: 修改的因果图
        :rtype: CausalGraph的实例

    .. py:method:: remove_edge(edge, observed=True)
        
        移除CausalGraph中的边。如果观察到，移除未观察到的潜在的混淆弧线。

        :param tuple edge: 2个元素分别表示边的起点和终点。
        :param bool, default=True observed: 如果未观察到，移除未观察到的潜在的混淆弧线。

    .. py:method:: remove_edges_from(edge_list, new=False, observed=True)
        
        移除图中在edge_list中所有的边。

        :param list edge_list: 要移除的边的列表。
        :param bool, default=False new: 如果new为真, 创建一个新的CausalGraph并移除边。
        :param bool, default=True observed: 如果未观察到，移除未观察到的潜在的混淆弧线。

        :returns: 修改的因果图
        :rtype: CausalGraph的实例

    .. py:method:: build_sub_graph(subset)
        
        返回一个新的CausalGraph作为图的子图，子图中的节点为subset中的节点。

        :param set subset: 子图的集合。

        :returns: 修改的因果图
        :rtype: CausalGraph的实例

    .. py:method:: remove_incoming_edges(x, new=False)
        
        移除x中所有节点的入射边。如果new为真，在新的CausalGraph做这个操作。

        :param set or list x:
        :param bool, default=False, new: 如果为真，返回一个新图。

        :returns: 修改的因果图
        :rtype: CausalGraph的实例

    .. py:method:: remove_outgoing_edges(x, new=False)
        
        移除x中所有节点的出射边。如果new为真，在新的CausalGraph做这个操作。

        :param set or list x:
        :param bool, default=False, new: 如果为真，返回一个新图。

        :returns: 修改的因果图
        :rtype: CausalGraph的实例

    .. py:property:: c_components
        
        图的C-components集合。
        
        :returns: 图的C-components集合
        :rtype: str的集合

    .. py:property:: observed_dag
        
        返回图的观测到的部分，包含观测到的节点和它们之间的边。
        
        :returns: 图的观测到的部分
        :rtype: networkx.MultiGraph

    .. py:property:: explicit_unob_var_dag
        
        构建一个新的DAG其中所有未观测到的混淆曲线由明确的未观测的变量取代。
        
        :returns: 有明确的未观测到节点的DAG
        :rtype: networkx.MultiGraph   
    
    .. py:property:: topo_order

        返回观测到的图中的节点的拓扑顺序。
        
        :returns: Nodes in the topological order
        :rtype: generator          