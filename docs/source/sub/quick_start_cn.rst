***********
快速开始
***********

在这部分中，我们首先展示几个简单的YLearn的用法示例。这些例子包含了大部分常见的功能。之后我们以一个使用 :class:`Why` 的例子来学习揭露数据中隐藏的因果关系。

示例用法
==============

我们在这部分中展示几个必要的YLearn的用法示例。细节请参考它们的具体的文档。

1. 因果图的表示

   对一个给定的因果图 :math:`X \leftarrow W \rightarrow Y`, 该因果图由 :class:`CausalGraph` 表示

    .. code-block:: python

        causation = {'X': ['W'], 'W':[], 'Y':['W']}
        cg = CausalGraph(causation=causation)

   :py:attr:`cg` 将成为YLearn中的因果图的表示.

2. 识别因果效应

   假如我们对识别因果估计量感兴趣 :math:`P(Y|do(X=x))` 在因果图 `cg` 中, 接着我们应该定义一个实例 :class:`CausalModel` 的实例并使用 :py:func:`identify()` 方法:

    .. code-block:: python

        cm = CausalModel(causal_graph=cg)
        cm.identify(treatment={'X'}, outcome={'Y'}, identify_method=('backdoor', 'simple'))

3. 估计因果效应

   通过 :class:`EstimatorModel` 估计因果效应由4步组成:

    * 以 :class:`pandas.DataFrame` 的形式给出数据, 找到 `treatment, outcome, adjustment, covariate` 的名称。
    * 使用 :class:`EstimatorModel` 的 :py:func:`fit()` 方法来训练模型。
    * 使用 :class:`EstimatorModel` 的 :py:func:`estimate()` 方法来估计测试数据中的因果效应。


案例分析
==========