.. _ce_int:

*************
CEInterpreter
*************

对于由一个估计模型估计的CATE :math:`\tau(v)` ，比如， 双机器学习模型， :class:`CEInterpreter` 解释
由构建的一个用于建模 :math:`\tau(v)` 和谐变量 :math:`v` 之间关系的决策树得到的结果。然后能够使用拟合树模型的决策规则
来分析 :math:`\tau(v)`。

类结构
================

.. py:class:: ylearn.effect_interpreter.ce_interpreter.CEInterpreter(*, criterion='squared_error', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=2022, max_leaf_nodes=None, max_features=None, min_impurity_decrease=0.0, min_weight_fraction_leaf=0.0, ccp_alpha=0.0, categories='auto')

    :param {"squared_error", "friedman_mse", "absolute_error",  "poisson"}, default="squared_error" criterion: 用于测量划分的质量的函数。
        支持的准则为"squared_error"用于均方误差，其等价于把方差减少作为特征选择的准则并且使用每一个终端节点的平均值最小化L2误差。"friedman_mse"为
        潜在的划分用均方误差和Friedman的改进分数。"absolute_error"用于平均绝对误差，其通过使用每一个终端节点的中位数最小化L1误差。"poisson"使用减少泊松偏差来寻找划分。
    :param {"best", "random"}, default="best" splitter: 用于在每个节点选择划分的策略。支持的策略"best"用于选择最佳划分，"random"用于选择最佳随机划分。
    :param int, default=None max_depth: 树的最大深度。如果为None，那么节点可以一直扩展直到所有的叶子都是纯的或者所有的叶子都包含小于min_samples_split个样本。
    :param int or float, default=2 min_samples_split: 划分内部节点所需要的最小的样本数量：
        - 如果是int，那么考虑 `min_samples_split` 为最小数量。
        - 如果是float, 那么 `min_samples_split` 是一个分数并且 `ceil(min_samples_split * n_samples)` 是对于每一个划分最小的样本数量。
    :param int or float, default=1 min_samples_leaf: 在一个叶子节点需要的最小的样本数量。
        一个在任意深度的划分点仅当它留下至少 ``min_samples_leaf`` 训练样本在它的左右分支时会被考虑。这可能有平滑模型的作用，尤其是在回归中。

            - 如果是int, 那么考虑 `min_samples_leaf` 为最小数量。
            - 如果是float, 那么 `min_samples_leaf` 是一个分数并且 `ceil(min_samples_leaf * n_samples)` 是对于每一个节点最小的样本数量。

    :param float, default=0.0 min_weight_fraction_leaf: 在一个叶子节点需要的所有权重总和（所有的输入样本）的最小加权分数。如果sample_weight没有被提供时，样本具有同样的权重。
    :param int, float or {"sqrt", "log2"}, default=None max_features: 寻找最佳划分时需要考虑的特征数量：

            - 如果是int，那么考虑 `max_features` 个特征在每个划分。
            - 如果是float，那么 `max_features` 是一个分数并且 `int(max_features * n_features)` 个特征在每个划分被考虑。
            - 如果是"sqrt"，那么 `max_features=sqrt(n_features)` 。
            - 如果是"log2"，那么 `max_features=log2(n_features)` 。
            - 如果是None，那么 `max_features=n_features` 。

    :param int random_state: 控制估计器的随机性。
    :param int, default to None max_leaf_nodes: 以最佳优先的方式使用 ``max_leaf_nodes`` 生成一棵树。
        最佳节点被定义为杂质相对减少。
        如果是None，那么叶子节点的数量没有限制。
    :param float, default=0.0 min_impurity_decrease: 一个节点将会被划分如果这个划分引起杂质的减少大于或者等于这个值。
        加权的杂质减少方程如下

            N_t / N * (impurity - N_t_R / N_t * right_impurity - N_t_L / N_t * left_impurity)

        其中 ``N`` 是样本的总数， ``N_t`` 是当前节点的样本数量， ``N_t_L`` 是左孩子节点的样本数量，并且 ``N_t_R`` 是右孩子节点的样本数量。
        ``N``, ``N_t``, ``N_t_R`` 以及 ``N_t_L`` 全都是指加权和，如果 ``sample_weight`` 被传入。

    .. py:method:: fit(data, est_model, **kwargs)

        拟合CEInterpreter模型来解释基于数据和est_model估计的因果效应。

        :param pandas.DataFrame data: 输入样本，用于est_model估计因果效应和用于CEInterpreter拟合。
        :param estimator_model est_model: est_model应该为ylearn的任何合理的估计器模型且已经拟合过了并且能够估计CATE。

        :returns: Fitted CEInterpreter
        :rtype: CEInterpreter的实例

    .. py:method:: interpret(*, v=None, data=None)

        在测试数据中解释拟合的模型。

        :param numpy.ndarray, optional, default=None v: ndarray形式的测试协变量。如果这被给出，那么数据将会被忽略并且模型会使用这个作为测试数据。
        :param pandas.DataFrame, optional, default=None data: DataFrame形式的测试数据。模型将仅使用这个如果v被设置为None。在这种情况下，如果数据也是None，那么训练的数据将会被使用。

        :returns: 对所有的样例解释的结果。
        :rtype: dict

    .. py:method:: plot(*, feature_names=None, max_depth=None, class_names=None, label='all', filled=False, node_ids=False, proportion=False, rounded=False, precision=3, ax=None, fontsize=None)

        绘制拟合的树模型。
        显示的样本计数由任何的可能存在的sample_weights加权。
        可视化自动适应轴的大小。
        使用 ``plt.figure`` 的 ``figsize`` 或者 ``dpi`` 参数来控制生成的大小。

        :returns: List containing the artists for the annotation boxes making up the
            tree.
        :rtype: annotations : list of artists

.. topic:: 例子

    pass