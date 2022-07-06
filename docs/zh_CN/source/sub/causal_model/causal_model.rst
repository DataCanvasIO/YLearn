.. _causal_model:

************
因果模型
************

:class:`CausalModel` 是一个核心对象来执行 :ref:`identification` 和寻找工具变量。

在介绍因果模型之前，我们首先需要阐明 **干涉** 的定义。干涉是对整个人群的，并给每一个人一些操作。
[Pearl]_ 定义了 :math:`do`-operator 来描述了这样的操作。概率模型不能够服务预测干涉效果，这导致了对因果模型的需求。

对 **causal model** 的正式定义归因于 [Pearl]_ 。一个因果模型是一个三元组

.. math::
    
    M = \left< U, V, F\right>

其中

* :math:`U` 是 **外生的** （由模型外的因素决定的变量）；
* :math:`V` 是 **内生的** 决定于 :math:`U \cup V`, 和 :math:`F` 是这样的一组函数

.. math::
        
        V_i = F_i(pa_i, U_i)

其中 :math:`pa_i \subset V \backslash V_i`.

例如， :math:`M = \left< U, V, F\right>` 是一个因果模型其中

.. math::
    
    V = \{V_1, V_2\}, 
    
    U = \{ U_1, U_2, I, J\},
    
    F = \{F_1, F_2 \}

这样

.. math::

    V_1 = F_1(I, U_1) = \theta_1 I + U_1\\
    V_2 = F_2(V_1, J, U_2, ) = \phi V_1 + \theta_2 J + U_2.

注意每个因果模型都可以和一个DAG关联并编码变量之间必要的因果关系信息。
YLearn使用 :class:`CausalModel` 来表示一个因果模型并支持许多关于因果模型的操作比如 :ref:`identification` 。

.. _identification:

识别
==============

为了表征干涉的效果，需要考虑 **因果效应** ，其是一个因果估计量包含 :math:`do`-operator 。把因果效应转变为对应的统计估计量的行为被称为 :ref:`identification` 且
在YLearn中的 :class:`CausalModel` 里实现。注意不是所有的因果效应都能被转变为统计估计量的。我们把这样的因果效应称为不可识别的。我们列出几个 `CausalModel` 支持的识别方法。

.. topic:: 后门调整

    :math:`X` 对 :math:`Y` 的因果效应由下式给出

    .. math::

        P(y|do(x)) = \sum_w P(y|x, w)P(w)
    
   如果变量的集合 :math:`W` 满足后门准则关于 :math:`(X, Y)`.

.. topic:: 前门调整

    :math:`X` 对 :math:`Y` 的因果效应由下式给出

    .. math::

        P(y|do(x)) = \sum_w P(w|x) \sum_{x'}P(y|x', w)P(x')
    
    如果变量的集合 :math:`W` 满足前门准则关于 :math:`(X, Y)` and if
    :math:`P(x, w) > 0`.

.. topic:: 通用识别

    [Shpitser2006]_ 给出一个必要充分的图的条件这样任意一个集合的变量对另一个任意集合的因果效应能够被唯一的识别无论它是不是可识别的。
    我们称验证这个条件对应的行动为 **通用识别**。

.. topic:: 找到工具变量

    当有未观测到的 :math:`X` 和 :math:`Y` 的混淆因素时，工具变量在识别和估计 :math:`X` 对 :math:`Y` 的因果效应很有用处。
    一组变量 :math:`Z` 被称为一组 **工具变量** 如果 :math:`Z` 中任意的 :math:`z` ：
    
    1. :math:`z` 对 :math:`X` 有因果效应。
    
    2. :math:`z` 对 :math:`Y` 的因果效应完全由于 :math:`X` 。
    
    3. 没有后门路径从 :math:`z` 到 :math:`Y` 。

.. topic:: 例子1: 用通用识别方法识别因果效应

    .. figure:: graph_un_arc.png
        
        因果结构，其中所有的未观察到的变量都被移除了，它们相关的因果关系都被混淆的弧线（有两个箭头的黑色虚线）取代了。
    
    对于图中的因果结构，我们想要使用 *通用识别* 方法识别 :math:`X` 对 :math:`Y` 的因果效应。第一步是用 :class:`CausalModel` 表示因果结构。
    
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

    然后我们需要为编码在 :py:attr:`cg` 中的因果结构定义一个 :class:`CausalModel` 的实例，从而进行识别。

    .. code-block:: python

        from ylearn.causal_model.model import CausalModel
        cm = CausalModel(causal_model=cg)
        stat_estimand = cm.id(y={'Y'}, x={'X'})
        stat_estimand.show_latex_expression()

    >>> :math:`\sum_{Z3, Z1, Z2}[P(Z2)P(Y|Z3, Z2)][P(Z1|Z2, X)][P(Z3|Z2)]`

    结果是想要的识别的在给定的因果结构中 :math:`X` 对 :math:`Y` 的因果效应。

.. topic:: 例子2: 使用后门调整识别因果效应

    .. figure:: backdoor.png

        所有节点都是观测到的变量。
    
    对于图中的因果结构，我们想要使用 *后门调整* 方法识别 :math:`X` 对 :math:`Y` 的因果效应。
    
    .. code-block:: python
        
        from ylearn.causal_model.graph import CausalGraph
        from ylearn.causal_model.model import CausalModel

        causation = {
            'X1': [], 
            'X2': [], 
            'X3': ['X1'], 
            'X4': ['X1', 'X2'], 
            'X5': ['X2'], 
            'X6': ['X'], 
            'X': ['X3', 'X4'], 
            'Y': ['X6', 'X4', 'X5', 'X'], 
        } 

        cg = CausalGraph(causation=causation)
        cm = CausalModel(causal_graph=cg)
        backdoor_set, prob = cm3.identify(treatment={'X'}, outcome={'Y'}, identify_method=('backdoor', 'simple'))['backdoor']

        print(backdoor_set)


    >>> ['X3', 'X4']

.. topic:: 例子3: 找到合理的工具变量

    .. figure:: iv1.png

        变量 :math:`p, t, l, g` 的因果结构

    我们想要为 :math:`t` 对 :math:`g` 的因果效应找到合理的工具变量。

    .. code-block:: python

        causation = {
            'p':[],
            't': ['p'],
            'l': ['p'],
            'g': ['t', 'l']
        }
        arc = [('t', 'g')]
        cg = CausalGraph(causation=causation, latent_confounding_arcs=arc)
        cm = CausalModel(causal_graph=cg)

        cm.get_iv('t', 'g')

    >>> No valid instrument variable has been found.

    .. figure:: iv2.png

        对变量 :math:`p, t, l, g` 的另一个因果结构

    我们依然想要在这个新的因果结构中，为 :math:`t` 对 :math:`g` 的因果效应找到合理的工具变量。

    .. code-block:: python

        causation = {
            'p':[],
            't': ['p', 'l'],
            'l': [],
            'g': ['t', 'l']
        }
        arc = [('t', 'g')]
        cg = CausalGraph(causation=causation, latent_confounding_arcs=arc)
        cm = CausalModel(causal_graph=cg)

        cm.get_iv('t', 'g')
    
    >>> {'p'}

类结构
================

.. py:class:: ylearn.causal_model.CausalModel(causal_graph=None, data=None)

    :param CausalGraph, optional, default=None causal_graph: CausalGraph的实例，编码了因果结构
    :param pandas.DataFrame, optional, default=None data: 用于发现因果结构的数据，如果causal_graph没有提供。

    .. py:method:: id(y, x, prob=None, graph=None)
        
        识别因果量 :math:`P(y|do(x))` 如果可识别否则返回 raise :class:`IdentificationError` 。
        注意这里我们仅考虑半马尔可夫因果模型，其中每个未观测到的变量正好是两个节点的父节点。这是因为任何的有未观测的变量的因果模型可以被转变为
        一个编码了同样集合的条件独立性的半马尔可夫模型。

        :param set of str y: 结果的名字的集合。
        :param set of str x: 治疗的名字的集合。
        :param Prob, optional, default=None prob: 编码在图中的概率分布。
        :param CausalGraph graph: CausalGraph编码了对应的因果结构中的信息。

        :returns: 转变的因果效应的概率分布。
        :rtype: Prob
        :raises IdentificationError: 如果感兴趣的因果效应不能识别，则raise IdentificationError。

    .. py:method:: is_valid_backdoor_set(set_, treatment, outcome)

        决定给定的集合是否是对结果的治疗的因果效应的一个合理的后门调整集合。

        :param set set_: 调整集合。
        :param set or list of str treatment: 治疗的名字。对单个治疗，str也是可以接受的。
        :param set or list of str outcome: 结果的名字。对单个结果，str也是可以接受的。

        :returns: True，如果在现在的因果图中，给定的集合是对结果的治疗的因果效应的一个合理的后门调整集合。
        :rtype: bool

    .. py:method::  get_backdoor_set(treatment, outcome, adjust='simple', print_info=False)
        
        对给定的治疗和结果返回后门调整集合。

        :param set or list of str treatment: 治疗的名字。对单个治疗，str也是可以接受的。
        :param set or list of str outcome: 结果的名字。对单个结果，str也是可以接受的。
        :param str adjust: 设置后门集合的样式。可选的选项是
                
                simple: 直接返回治疗的父节点集合
                
                minimal: 返回最小的后门调整集合
                
                all: 返回所有合理的后门调整集合。
        
        :param bool, default=False print_info: 如果为True，打印识别的结果。

        :returns: 第一个元素是调整列表，同时第二个是编码的Prob。
        :rtype: 两个元素的元组
        :raises IdentificationError: Raise error如果样式不在simple，minimal或者all或者没有集合能满足后门准则。

    .. py:method:: get_backdoor_path(treatment, outcome)

        返回所有的连接治疗和结果的后门路径。

        :param str treatment: 治疗的名字。
        :param str outcome: 结果的名字。

        :returns: 一个包含图中所有合理的治疗和结果之间的后门路径的列表。
        :rtype: list

    .. py:method:: has_collider(path, backdoor_path=True)

        如果现在图的path中有一个对撞，返回True，否则返回False。

        :param list of str path: 包含路径中节点的列表。
        :param bool, default=True backdoor_path: 该路径是否是一个后门路径。

        :returns: True，如果path有一个对撞。
        :rtype: bool

    .. py:method:: is_connected_backdoor_path(path)

        测试是否一个后门路径是连接的。

        :param list of str path: 描述这个路径的列表。

        :returns: True，如果路径是一个d-connected的后门路径，否则False。
        :rtype: bool

    .. py:method:: is_frontdoor_set(set_, treatment, outcome)

        决定给定的集合是否是对结果的治疗的因果效应的一个合理的前门调整集合。

        :param set set_: 等待决定是否是合理的前门调整集合的集合。
        :param str treatment: 治疗的名字。
        :param str outcome: 结果的名字。

        :returns: True如果给定的集合是对结果的治疗的因果效应的一个合理的前门调整集合。
        :rtype: bool

    .. py:method:: get_frontdoor_set(treatment, outcome, adjust='simple')

        返回用于调整治疗和结果之间因果效应的前门集合。

        :param set of str or str treatment: 治疗的名字。应该只包含一个元素。
        :param set of str or str outcome: 结果的名字。应该只包含一个元素。
        :param str, default='simple' adjust: 可选的选项包括
                'simple': 返回有最少数量元素的前门集合。
                
                'minimal': 返回有最少数量元素的前门集合。
                
                'all': 返回所有可能的前门集合。
        
        :returns: 2个元素（adjustment_set, Prob）
        :rtype: 元组
        :raises IdentificationError: Raise error如果样式不在simple，minimal或者all或者没有集合能满足前门准则。

    .. py:method:: get_iv(treatment, outcome)

        为结果的治疗的因果效应找到工具变量。

        :param iterable treatment: 治疗的名字（们）。
        :param iterable outcome: 结果的名字（们）。

        :returns: 一个合理的工具变量集合将会是空的如果没有这样的集合。
        :rtype: set

    .. py:method:: is_valid_iv(treatment, outcome, set_)

        决定给出的集合是否是一个合法的工具变量集合。

        :param iterable treatment: 治疗的名字（们）。
        :param iterable outcome: 结果的名字（们）。
        :param set set_: 等待测试的集合。

        :returns: True如果集合是一个合理的工具变量集合否则False。
        :rtype: bool

    .. py:method:: identify(treatment, outcome, identify_method='auto')
        
        识别因果效应表达式。识别是转变任何因果效应量的操作。比如，用do operator的量，变为对应的统计量这样它就可以用给出的数据估计因果效应。但是，
        注意不是所有的因果量都是可识别的，这种情况下，一个IdentificationError被抛出。

        :param set or list of str treatment: 治疗名字的集合。
        :param set or list of str outcome: 结果名字的集合。
        :param tuple of str or str, optional, default='auto' identify_method: 如果传入的值是元组或者列表，那么它应该有两个元素，
                其中第一个是识别方法，第二个是返回的集合样式。

                可选的选项：
                
                    'auto' : 使用所有可能的方法进行识别
                    
                    'general': 通用识别方法，看id()
                    
                    *('backdoor', 'simple')*: 返回治疗和结果的所有的直接的混淆因素的集合作为后门调整集合。
                    
                    *('backdoor', 'minimal')*: 返回所有的可能的有最小数量元素的后门调整集合。
                    
                    *('backdoor', 'all')*: 返回所有的可能的后门调整集合。
                    
                    *('frontdoor', 'simple')*: 返回所有的可能的有最小数量元素的前门调整集合。
                    
                    *('frontdoor', 'minimal')*: 返回所有的可能的有最小数量元素的前门调整集合。
                    
                    *('frontdoor', 'all')*: 返回所有的可能的前门调整集合。

        :returns: 一个python字典，其中字典中的键是识别方法，值是对应的结果。
        :rtype: dict
        :raises IdentificationError: 如果因果效应不可识别或者identify_method给的不正确。

    .. py:method:: estimate(estimator_model, data=None, *, treatment=None, outcome=None, adjustment=None, covariate=None, quantity=None, **kwargs)

        估计新的数据集中识别的因果效应。

        :param EstimatorModel estimator_model: 任何在EstimatorModel中实现的合适的估计器模型可以在这里使用。
        :param pandas.DataFrame, optional, default=None data: 用于估计的因果效应的数据集。如果是None，使用用于因果图发现的数据。
        :param  set or list, optional, default=None treatment: 治疗的名字们。如果是None，用于后门调整的治疗被当作治疗。
        :param set or list, optional, default=None outcome: 结果的名字们。如果是None，用于后门调整的结果被当作结果。
        :param set or list, optional, default=None adjustment: 调整集合的名字们。如果是None，调整集合由CausalModel找到的最简单的后门集合给出。
        :param set or list, optional, default=None covariate: 协变量集合的名字。如果是None则忽略。
        :param str, optional, default=None quantity: 估计因果效应时，感兴趣的量。

        :returns: 估计的数据中的因果效应。
        :rtype: np.ndarray or float

    .. py:method:: identify_estimate(data, outcome, treatment, estimator_model=None, quantity=None, identify_method='auto', **kwargs)

        组合识别方法和估计方法。然而，既然现在实现的估计器模型自动假设（有条件地）无混淆（除了有关iv的方法）。我们可能仅考虑使用后门集合调整来实现无混淆条件。

        :param set or list of str, optional treatment: 治疗的名字们。
        :param set or list of str, optional outcome: 结果的名字们。
        :param tuple of str or str, optional, default='auto' identify_method: 如果传入的值是元组或者列表，那么它应该有两个元素，
                其中第一个是识别方法，第二个是返回的集合样式。

                可选的选项：
                
                    'auto' : 使用所有可能的方法进行识别
                    
                    'general': 通用识别方法，看id()
                    
                    *('backdoor', 'simple')*: 返回治疗和结果的所有的直接的混淆因素的集合作为后门调整集合。
                    
                    *('backdoor', 'minimal')*: 返回所有的可能的有最小数量元素的后门调整集合。
                    
                    *('backdoor', 'all')*: 返回所有的可能的后门调整集合。
                    
                    *('frontdoor', 'simple')*: 返回所有的可能的有最小数量元素的前门调整集合。
                    
                    *('frontdoor', 'minimal')*: 返回所有的可能的有最小数量元素的前门调整集合。
                    
                    *('frontdoor', 'all')*: 返回所有的可能的前门调整集合。
        
        :param str, optional, default=None quantity: 估计因果效应时，感兴趣的量。

        :returns: 估计的数据中的因果效应。
        :rtype: np.ndarray or float
