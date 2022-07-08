***********
因果树
***********

因果树是一个数据驱动的方法，用来把数据划分为因果效应幅度不同的亚群 [Athey2015]_ 。这个方法在给定调整集（协变量） :math:`V` 无混淆满足时适用。
感兴趣的因果效应是CATE：

.. math::

    \tau(v) := \mathbb{}[Y_i(do(X=x_t)) - Y_i(do(X=x_0)) | V_i = v]

因为事实上反事实无法被观测到， [Athey2015]_ 开发了一个诚实的方法，其中损失函数（构建树的准则）被设计为

.. math::

    e (S_{tr}, \Pi) := \frac{1}{N_{tr}} \sum_{i \in S_{tr}} \hat{\tau}^2 (V_i; S_{tr}, \Pi) - \frac{2}{N_{tr}} \cdot \sum_{\ell \in \Pi} \left( \frac{\Sigma^2_{S_{tr}^{treat}}(\ell)}{p} + \frac{\Sigma^2_{S_{tr}^{control}}(\ell)}{1 - p}\right)

其中 :math:`N_{tr}` 是训练集 :math:`S_{tr}` 中样本的数量, :math:`p` 是训练集中治疗组和控制组样本个数的比，且

.. math::

    \hat{\tau}(v) = \frac{1}{\#(\{i\in S_{treat}: V_i \in \ell(v; \Pi)\})} \sum_{ \{i\in S_{treat}: V_i \in \ell(v; \Pi)\}} Y_i \\
    - \frac{1}{\#(\{i\in S_{control}: V_i \in \ell(v; \Pi)\})} \sum_{ \{i\in S_{control}: V_i \in \ell(v; \Pi)\}} Y_i.

.. topic:: 例子

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt

        from ylearn.estimator_model.causal_tree import CausalTree
        from ylearn.exp_dataset.exp_data import sq_data
        from ylearn.utils._common import to_df


        # build dataset
        n = 2000
        d = 10     
        n_x = 1
        y, x, v = sq_data(n, d, n_x)
        true_te = lambda X: np.hstack([X[:, [0]]**2 + 1, np.ones((X.shape[0], n_x - 1))])
        data = to_df(treatment=x, outcome=y, v=v)
        outcome = 'outcome'
        treatment = 'treatment'
        adjustment = data.columns[2:]

        # build test data
        v_test = v[:min(100, n)].copy()
        v_test[:, 0] = np.linspace(np.percentile(v[:, 0], 1), np.percentile(v[:, 0], 99), min(100, n))
        test_data = to_df(v=v_test)
    
    训练 `CausalTree` 并在测试数据中使用它：

    .. code-block:: python

        ct = CausalTree(min_samples_leaf=3, max_depth=5)
        ct.fit(data=data, outcome=outcome, treatment=treatment, adjustment=adjustment)
        ct_pred = ct.estimate(data=test_data)


类结构
================

.. py:class:: ylearn.estimator_model.causal_tree.CausalTree(*, splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=2022, max_leaf_nodes=None, max_features=None, min_impurity_decrease=0.0, min_weight_fraction_leaf=0.0, ccp_alpha=0.0, categories='auto')

    :param {"best", "random"}, default="best" splitter: 用于选择每个节点划分的策略。支持的策略为选择最佳划分的"best"和选择最佳随机划分的"random"。
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

    :param str, optional, default='auto' categories: 

    .. py:method:: fit(data, outcome, treatment, adjustment=None, covariate=None, treat=None, control=None)
        
        基于数据拟合模型来估计因果效应。

        :param pandas.DataFrame data: 输入样本，用于est_model估计因果效应和用于CEInterpreter拟合。
        :param list of str, optional outcome: 结果的名字。
        :param list of str, optional treatment: 治疗的名字。
        :param list of str, optional, default=None covariate: 协变量向量的名字。
        :param list of str, optional, default=None adjustment: 协变量向量的名字。注意我们可能只需要协变量集合，其通常是调整集合的一个子集。
        :param int or list, optional, default=None treat: 如果只有一个离散的治疗，那么treat表示治疗组。如果有多个治疗组，
            那么treat应该是一个str列表，其长度等于治疗的数量。比如，当有多个离散的治疗时，

                array(['run', 'read'])

            意味着第一个治疗的治疗值为 'run' 且第二个治疗为 'read'。
        :param int or list, optional, default=None control: 参考treat。
        
        :returns: 拟合的CausalTree
        :rtype: CausalTree的实例

    .. py:method:: estimate(data=None, quantity=None)

        估计数据中治疗对结果的因果效应。

        :param pandas.DataFrame, optional, default=None data: 如果为None，数据将被设置为训练数据。
        :param str, optional, default=None quantity: 返回的估计结果的选项。量的可能值包括：
                
                1. *'CATE'* : 估计器将会估计CATE；
                
                2. *'ATE'* : 估计器将会估计ATE；
                
                3. *None* : 估计器将会估计ITE或CITE。

        :returns: 量类型的估计的因果效应。
        :rtype: ndarray or float, optional

    .. py:method:: plot_causal_tree(feature_names=None, max_depth=None, class_names=None, label='all', filled=False, node_ids=False, proportion=False, rounded=False, precision=3, ax=None, fontsize=None)

        绘制策略树。
        显示的样本计数由任何的可能存在的sample_weights加权。
        可视化自动适应轴的大小。
        使用 ``plt.figure`` 的 ``figsize`` 或者 ``dpi`` 参数来控制生成的大小。

        :returns: List containing the artists for the annotation boxes making up the
            tree.
        :rtype: annotations : list of artists
    
    .. py:method:: decision_path(*, data=None, wv=None)

        返回决策路径。

        :param numpy.ndarray, default=None wv: 输入样本是一个ndarray。 如果是None，那么DataFrame的数据将会被用作输入样本。
        :param pandas.DataFrame, default=None data: 输入样本。数据必须包含用于训练模型的协变量的列。如果为None，训练数据将会被传入作为输入样本。

        :returns: Return a node indicator CSR matrix，其中非零元素表示穿过节点的样本。
        :rtype: indicator : shape为(n_samples, n_nodes)的稀疏矩阵

    .. py:method:: apply(*, data=None, wv=None)

        返回每个样本被预测为的叶子的索引。
        
        :param numpy.ndarray, default=None wv: 输入样本是一个ndarray。 如果是None，那么DataFrame的数据将会被用作输入样本。
        :param pandas.DataFrame, default=None data: 输入样本。数据必须包含用于训练模型的协变量的列。如果为None，训练数据将会被传入作为输入样本。

        :returns: 对于v中每个数据点v_i，返回v_i结束在的叶子的索引。叶子在 ``[0; self.tree_.node_count)`` 中编号，可能编号有间隙。
        :rtype: v_leaves : array-like of shape (n_samples, )

    .. py:property:: feature_importance

        :returns: 按特征的归一化的总减少标准(Gini importance)。
        :rtype: ndarray of shape (n_features,)