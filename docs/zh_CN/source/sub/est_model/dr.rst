*************
双鲁棒
*************

双鲁棒方法（参考 [Funk2010]_ ）估计因果效应当治疗是离散的且满足无混淆条件。
训练一个双鲁棒模型由3步组成。

1. 令 :math:`k` 为一个int。形成一个对数据 :math:`\{(X_i, W_i, V_i, Y_i)\}_{i = 1}^n` 的 :math:`K`-fold 随机划分，这样

   .. math::

        \{(x_i, w_i, v_i, y_i)\}_{i = 1}^n = D_k \cup T_k

   其中 :math:`D_k` 表示训练数据且 :math:`T_k` 表示测试数据且 :math:`\cup_{k = 1}^K T_k = \{(X_i, W_i, V_i, Y_i)\}_{i = 1}^n`.

2. 对于每个 :math:`k`, 训练两个模型 :math:`f(X, W, V)` 和 :math:`g(W, V)` 在 :math:`D_k` 上来分别预测 :math:`y` 和 :math:`x`。接着估计它们在 :math:`T_k` 中的性能，
   结果保存为 :math:`\{(\hat{X}, \hat{Y})\}_k` 。所有的 :math:`\{(\hat{X}, \hat{Y})\}_k` 将被合并来给出新的数据集 :math:`\{(\hat{X}_i, \hat{Y}_i(X, W, V))\}_{i = 1}^n` 。

3. 对于任何给定的一对治疗组其中 :math:`X=x` 和控制组其中 :math:`X = x_0` ，我们构建最终数据集 :math:`\{(V, \tilde{Y}_x - \tilde{Y}_0)\}` 其中 :math:`\tilde{Y}_x`
   被定义为

   .. math::

        \tilde{Y}_x & = \hat{Y}(X=x, W, V) + \frac{(Y - \hat{Y}(X=x, W, V)) * \mathbb{I}(X=x)}{P[X=x| W, V]} \\
        \tilde{Y}_0 & = \hat{Y}(X=x_0, W, V) + \frac{(Y - \hat{Y}(X=x_0, W, V)) * \mathbb{I}(X=x_0)}{P[X=x_0| W, V]}
    
   并在这个数据集上训练最终的机器学习模型 :math:`h(W, V)` 来预测因果效应 :math:`\tau(V)`

   .. math::

       \tau(V) =  \tilde{Y}_x - \tilde{Y}_0 = h(V).
    
   接着我们可以直接估计因果效应，通过传入协变量 :math:`V` 到模型 :math:`h(V)` 。


.. topic:: 例子

    .. code-block:: python

        import numpy as np
        from numpy.random import multivariate_normal
        
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
        
        import matplotlib.pyplot as plt

        from ylearn.estimator_model.meta_learner import SLearner, TLearner, XLearner
        from ylearn.estimator_model.doubly_robust import DoublyRobust
        from ylearn.exp_dataset.exp_data import binary_data
        from ylearn.utils import to_df

        # build the dataset
        d = 5
        n = 2500
        n_test = 250

        y, x, w = binary_data(n=n, d=d, n_test=n_test)
        data = to_df(outcome=y, treatment=x, w=w)
        outcome = 'outcome'
        treatment = 'treatment'
        adjustment = data.columns[2:]

        # build the test dataset
        treatment_effect = lambda x: (1 if x[1] > 0.1 else 0) * 8

        w_test = multivariate_normal(np.zeros(d), np.diag(np.ones(d)), n_test)
        delta = 6/n_test
        w_test[:, 1] = np.arange(-3, 3, delta)

    训练 `DoublyRobust` 模型。
    
    .. code-block:: python

        dr = DoublyRobust(
            x_model=RandomForestClassifier(n_estimators=100, max_depth=100, min_samples_leaf=int(n/100)),
            y_model=GradientBoostingRegressor(n_estimators=100, max_depth=100, min_samples_leaf=int(n/100)),
            yx_model=GradientBoostingRegressor(n_estimators=100, max_depth=100, min_samples_leaf=int(n/100)),
            cf_fold=1, 
            random_state=2022,
        )
        dr.fit(data=data, outcome=outcome, treatment=treatment, covariate=adjustment,)
        dr_pred = dr.estimate(data=test_data, quantity=None).squeeze()


类结构
================

.. py:class:: ylearn.estimator_model.doubly_robust.DoublyRobust(x_model, y_model, yx_model, cf_fold=1, random_state=2022, categories='auto')

    :param estimator, optional x_model: 经过训练的机器学习模型，用于对治疗建模。任何合理的x_model应该实现 :py:func:`fit()` 和 :py:func:`predict_proba()` 方法。
    :param estimator, optional y_model: 经过训练的机器学习模型，用于使用协变量（可能是调整）和治疗对结果建模。任何合理的y_model应该实现 :py:func:`fit()` 和 :py:func:`predict()` 方法。
    :param estimator, optional yx_model: 经过在双鲁棒方法的最后阶段训练的机器学习模型，用于使用协变量（可能是调整）对因果效应建模。任何合理的yx_model应该实现 :py:func:`fit()` 和 :py:func:`predict()` 方法。
    
    :param int, default=1 cf_fold: 在第一阶段执行交叉拟合的折的数量。
    :param int, default=2022 random_state:
    :param str, optional, default='auto' categories:
    
    .. py:method:: fit(data, outcome, treatment, adjustment=None, covariate=None, treat=None, control=None, combined_treatment=True, **kwargs)
        
        拟合DoublyRobust估计器模型。注意训练一个双鲁棒模型有三个阶段，其中我们在 :py:func:`_fit_1st_stage` 和 :py:func:`_fit_2nd_stage` 中实现它们。

        :param pandas.DataFrame data: 训练估计器的训练数据集。
        :param list of str, optional outcome: 结果的名字。
        :param list of str, optional treatment: 治疗的名字。
        :param list of str, optional, default=None adjustment: 保证无混淆的调整集的名字。
        :param list of str, optional, default=None covariate: 协变量的名字。
        :param int, optional treat: 预期治疗组的标签。如果为None，那么 :py:attr:`treat` 将会被设置为1。
            在单个离散治疗的情况下，treat应该是所有可能的治疗值之一的int或者str，
            其表示预期的治疗值，在有多个离散治疗的情况下，treat应该是一个列表或者ndarray，其中treat[i]表示第i个预期的治疗值。例如，
            当有多个离散治疗，array(['run', 'read'])意味着第一个治疗的治疗值是 'run' ，第二个治疗是 'read' 。
        :param int, optional control: 预期控制组的标签。这和treat的情况相似。如果是None，那么 :py:attr:`control` 将会被设置为0。


        :returns: 拟合的DoublyRobust的实例。
        :rtype: DoublyRobust的实例

    .. py:method:: estimate(data=None, quantity=None, treat=None, all_tr_effects=False)
        
        用量的类型估计因果效应。

        :param pandas.DataFrame, optional, default=None data: 测试数据。注意被设置为None，模型会使用训练数据。
        :param str, optional, default=None quantity: 返回的估计结果的选项。量的可能值包括：
                
                1. *'CATE'* : 估计器将会估计CATE；
                
                2. *'ATE'* : 估计器将会估计ATE；
                
                3. *None* : 估计器将会估计ITE或CITE。
        :param float or numpy.ndarray, optional, default=None treat: 在单个离散治疗的情况下，treat应该是所有可能的治疗值之一的int或者str，
            其表示预期的治疗值，在有多个离散治疗的情况下，treat应该是一个列表或者ndarray，其中treat[i]表示第i个预期的治疗值。例如，
            当有多个离散治疗，array(['run', 'read'])意味着第一个治疗的治疗值是 'run' ，第二个治疗是 'read' 。
        :param bool, default=False, all_tr_effects: 如果为True，返回所有的因果效应和所有的 :py:attr:`treatments` 的值，否则，仅返回
            在如果提供了的 :py:attr:`treat` 中的治疗的因果效应。如果 :py:attr:`treat` 没提供，那么治疗的值作为拟合估计器模型的值。

        :returns: 估计的因果效应
        :rtype: ndarray

    .. py:method:: effect_nji(data=None)
        
        用不同的治疗值计算因果效应。注意这个方法仅将把任何有离散治疗的问题转变为二元治疗。能够使用 :py:func:`_effect_nji_all` 去获得 :py:attr:`treatment` 取
        :py:attr:`treat` 所有值时的因果效应。

        :returns: 不同治疗值的因果效应。
        :rtype: ndarray

    .. py:method:: comp_transormer(x, categories='auto')
        
        把离散的治疗正确转变为独热向量。

        :param numpy.ndarray, shape (n, x_d) x:  包含治疗变量信息的数组。
        :param str or list, optional, default='auto' categories:

        :returns: 转变的独热向量。
        :rtype: numpy.ndarray
