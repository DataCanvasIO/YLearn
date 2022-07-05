.. _policy_int:

*****************
PolicyInterpreter
*****************

:class:`PolicyInterpreter` 能够被用于解释由一个实例 :class:`PolicyTree` 返回的策略。通过给不同的样例分配不同的策略，
它的目标是最大化一个子群的因果效应并把它们与那些有负的因果效应的分开。

.. topic:: 例子

    我们构建了一个数据集，其中给定协变量 :math:`v` 和二元的治疗方案 :math:`x`， 接受治疗的因果效应 :math:`y` 是正的如果 :math:`v` 的第一个纬度是正的，反之是负的。
    `PolicyInterpreter` 的目标是帮助每个个体决定是否采取治疗，等价于因果效应是否是正的。

    .. code-block:: python

        import numpy as np
        from ylearn.utils import to_df

        # build dataset
        v = np.random.normal(size=(1000, 10))
        y = np.hstack([v[:, [0]] < 0, v[:, [0]] > 0])

        data = to_df(v=v)
        covariate = data.columns

        # train the `PolicyInterpreter`
        from ylearn.effect_interpreter.policy_interpreter import PolicyInterpreter
        pit = PolicyInterpreter(max_depth=2)
        pit.fit(data=data, est_model=None, covariate=covariate, effect_array=y.astype(float))

        pit_result = pit.interpret()

    >>> 06-02 17:06:49 I ylearn.p.policy_model.py 448 - Start building the policy tree with criterion PRegCriteria
    >>> 06-02 17:06:49 I ylearn.p.policy_model.py 464 - Building the policy tree with splitter BestSplitter
    >>> 06-02 17:06:49 I ylearn.p.policy_model.py 507 - Building the policy tree with builder DepthFirstTreeBuilder

    解释的结果:

    .. code-block:: python

        for i in range(57, 60):
            print(f'the policy for the sample {i}\n --------------\n' + pit_result[f'sample_{i}'] + '\n')

    >>> the policy for the sample 57
    >>> --------------
    >>> decision node 0: (covariate [57, 0] = -0.0948629081249237) <= 8.582111331634223e-05
    >>> decision node 1: (covariate [57, 8] = 1.044342041015625) > -2.3793461322784424
    >>> The recommended policy is treatment 0 with value 1.0

    >>> the policy for the sample 58
    >>> --------------
    >>> decision node 0: (covariate [58, 0] = 0.706959068775177) > 8.582111331634223e-05
    >>> decision node 4: (covariate [58, 5] = 0.9160318374633789) > -2.575441598892212
    >>> The recommended policy is treatment 1 with value 1.0

类结构
================

.. py:class:: ylearn.interpreter.policy_interpreter.PolicyInterpreter(*, criterion='policy_reg', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=2022, max_leaf_nodes=None, max_features=None, min_impurity_decrease=0.0, ccp_alpha=0.0, min_weight_fraction_leaf=0.0)

    :param {'policy_reg'}, default="'policy_reg'" criterion: 该函数用于测量划分的质量。训练树的准则是(使用Einstein notation)

            .. math::

                    S = \sum_i g_{ik} y^k_{i},

            其中， :math:`g_{ik} = \phi(v_i)_k` 是一个映射，从协变量， :math:`v_i`，到一个有且仅有一个非零元素的在 :math:`R^k` 空间中的基向量。
            通过使用这个准则，模型的目标是找到能够生成最大因果效应的治疗的索引，等价于找到最优的策略。
    :param {"best", "random"}, default="best" splitter: 用于选择每个节点划分的策略。支持的策略是"best"来选择最优划分和"random"来选择最优随机划分。
    :param int, default=None max_depth: 树的最大深度。如果为None，那么节点一直扩展直到所有的叶子都是纯的或者所有的叶子都包含小于min_samples_split个样本。
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

    .. py:method:: fit(data, est_model, *, covariate=None, effect=None, effect_array=None)

        拟合PolicyInterpreter模型来解释基于数据和est_model估计的因果效应的策略。

        :param pandas.DataFrame data: 输入样本，用于est_model估计因果效应和用于CEInterpreter拟合。
        :param estimator_model est_model: est_model应该为ylearn的任何合理的估计器模型且已经拟合过了并且能够估计CATE。
        :param list of str, optional, default=None covariate: 协变量的名字。
        :param list of str, optional, default=None effect: 在 `data` 中因果效应的名字。如果 `effect_array` 不是None，那么 `effect` 将会被忽略。
        :param numpy.ndarray, default=None effect_array: 等待被 :class:`PolicyInterpreter` 解释的因果效应。如果这没有被提供，那么 `effect` 不能是None.

        :returns: Fitted PolicyInterpreter
        :rtype: PolicyInterpreter的实例

    .. py:method:: interpret(*, data=None)

        在测试数据中解释拟合的模型。

        :param pandas.DataFrame, optional, default=None data: DataFrame形式的测试数据。模型将仅使用这个如果v被设置为None。在这种情况下，如果数据也是None，那么训练的数据将会被使用。

        :returns: 对所有的样例解释的结果。
        :rtype: dict

    .. py:method:: plot(*, feature_names=None, max_depth=None, class_names=None, label='all', filled=False, node_ids=False, proportion=False, rounded=False, precision=3, ax=None, fontsize=None)

        绘制树模型。
        显示的样本计数由任何的可能存在的sample_weights加权。
        可视化自动适应轴的大小。
        使用 ``plt.figure`` 的 ``figsize`` 或者 ``dpi`` 参数来控制生成的大小。

        :returns: List containing the artists for the annotation boxes making up the
            tree.
        :rtype: annotations : list of artists
