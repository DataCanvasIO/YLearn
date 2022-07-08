*************************************
因果效应的近似边界
*************************************

许多的估计器模型需要通常无法测试的无混淆条件。一个适用的方法是，在深入具体的估计之前，构建我们的因果效应的上下界。

YLearn中有四个不同的界。我们在下面简单介绍它们。细节请参考 [Neal2020]_ 。

.. topic:: 无假设界

    假设
    
    .. math::

        \forall x, a \leq Y(do(x)) \leq b,
    
    那么我们有

    .. math::

        \mathbb{E}[Y(do(1)) - Y(do(0))] & \leq \pi \mathbb{E}[Y|X = 1] + (1 - \pi) b - \pi a - (1 - \pi )\mathbb{E}[Y| X = 0]\\
        \mathbb{E}[Y(do(1)) - Y(do(0))] & \geq \pi \mathbb{E}[Y|X = 1] + (1 - \pi) a - \pi b - (1 - \pi )\mathbb{E}[Y| X = 0]

    其中 :math:`\pi` 是令 :math:`X=1` 的概率。

.. topic:: 非负单调治疗响应界

    假如
    
    .. math::

        \forall i, Y(do(1)) \geq Y(do(0)),
    
    其意味着 *治疗只会帮助* 。那么我们有下面的界：
    
    .. math::

        \mathbb{E}[Y(do(1)) - Y(do(0))] & \leq \pi \mathbb{E}[Y|X = 1] + (1 - \pi) b - \pi a - (1 - \pi )\mathbb{E}[Y| X = 0]\\
        \mathbb{E}[Y(do(1)) - Y(do(0))] & \geq 0

.. topic:: 非正单调治疗响应界

    假如
    
    .. math::

        \forall i, Y(do(1)) \leq Y(do(0)),
    
    其意味着 *治疗不会帮助* 。那么我们有下面的界：
    
    .. math::

        \mathbb{E}[Y(do(1)) - Y(do(0))] & \leq 0\\
        \mathbb{E}[Y(do(1)) - Y(do(0))] & \geq \pi \mathbb{E}[Y|X = 1] + (1 - \pi) a - \pi b - (1 - \pi )\mathbb{E}[Y| X = 0].
    
.. topic:: 最优治疗选择界

    假如
    
    .. math::
        
        X = 1 &\implies Y(do(1)) \geq Y(do(0)) \\
        X = 0 & \implies Y(do(0)) \geq Y(do(1))            

    其意味着 *人们总是接受对他们而言最好的治疗* 。那么我们有下面的界：
    
    .. math::

        \mathbb{E}[Y(do(1)) - Y(do(0))] & \leq \pi \mathbb{E}[Y|X = 1] - \pi a\\
        \mathbb{E}[Y(do(1)) - Y(do(0))] & \geq (1 - \pi) a - (1 - \pi )\mathbb{E}[Y| X = 0].

    还有一种最优治疗选择界：

    .. math::

        \mathbb{E}[Y(do(1)) - Y(do(0))] & \leq \mathbb{E}[Y|X = 1] - \pi a - (1 - \pi)\mathbb{E}[Y|X=0]\\
        \mathbb{E}[Y(do(1)) - Y(do(0))] & \geq \pi\mathbb{E}[Y|X = 1] + (1 - \pi) a - \mathbb{E}[Y| X = 0].

.. topic:: 例子

    .. code-block:: python

        import numpy as np

        from ylearn.estimator_model.approximation_bound import ApproxBound
        from ylearn.exp_dataset.exp_data import meaningless_discrete_dataset_
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        data = meaningless_discrete_dataset_(num=num, confounder_n=3, treatment_effct=[2, 5, -8], random_seed=0)
        treatment = 'treatment'
        w = ['w_0', 'w_1', 'w_2']
        outcome = 'outcome'

        bound = ApproxBound(y_model=RandomForestRegressor(), x_model=RandomForestClassifier())
        bound.fit(data=data, treatment=treatment, outcome=outcome, covariate=w,)

    >>> ApproxBound(y_model=RandomForestRegressor(), x_prob=array([[0.  , 0.99, 0.01],
                [0.  , 0.99, 0.01],
                [1.  , 0.  , 0.  ],
                ...,
                [0.  , 1.  , 0.  ],
                [0.01, 0.99, 0.  ],
                [0.01, 0.99, 0.  ]]), x_model=RandomForestClassifier())
        
    .. code-block:: python
        
        b_l, b_u = bound1.estimate()
        b_l.mean()
    
    >>> -7.126728994957785

    .. code-block:: python

        b_u.mean()

    >>> 8.994011617037696

类结构
================

.. py:class:: ylearn.estimator_model.approximation_bound.ApproxBound(y_model, x_prob=None, x_model=None, random_state=2022, is_discrete_treatment=True, categories='auto')

    一个用于估计因果效应上下界的模型。

    :param estimator, optional y_model: 任何合理的y_model应该实现fit()和predict()方法。
    :param ndarray of shape (c, ), optional, default=None x_prob: 分配给x的对应值的概率数组，其中c是不同治疗类的数量。数组中所有元素都是
            正数且和为1。比如，x_prob =  array([0.5, 0.5])意味着 x = 0 和 x = 1 的概率为0.5。如果你使用多个治疗，请把这一项设置为None。
    :param estimator, optional, default=None x_model: 用于预测治疗概率的模型。任何合理的x_model应该实现fit()和predict_proba()方法。
    :param int, optional, default=2022 random_state:
    :param bool, optional, default=True is_discrete_treatment: True，如果治疗是离散的。
    :param str, optional, default='auto' categories:

    .. py:method:: fit(data, outcome, treatment, covariate=None, is_discrete_covariate=False, **kwargs)
        
        拟合 x_model 和 y_model.

        :param pandas.DataFrame data: 训练数据。
        :param list of str, optional outcome: 结果的名字。
        :param list of str, optional treatment: 治疗的名字。
        :param list of str, optional, default=None covariate: 协变量的名字。
        :param bool, optional, default=False is_discrete_covariate:

        :returns: ApproxBound的拟合的实例。
        :rtype: ApproxBound的实例。
        :raises ValueError:  当治疗不是离散的，Raise error。

    .. py:method:: estimate(data=None, treat=None, control=None, y_upper=None, y_lower=None, assump=None,)
        
        估计治疗对结果的因果效应的近似界。

        :param pandas.DataFrame, optional, default=None data: 测试数据。如果为None，模型将会使用训练数据。
        :param ndarray of str, optional, default=None treat: 治疗组的值。比如，当有多个离散的治疗时，array(['run', 'read'])意味着第一个治疗
                的治疗值为 'run'，第二个治疗是 'read'。
        :param ndarray of str, optional, default=None control: 控制组的值。
        :param float, defaults=None y_upper: 结果的上界。
        :param float, defaults=None y_lower: 结果的下界。
        :param str, optional, default='no-assump' assump: 返回界的选项。应该是其中之一
                
                1. *no-assump*: 计算无假设界，其结果总是包含0。
                
                2. *non-negative*: 治疗总是正的。
                
                3. *non-positive*: 治疗总是负的。
                
                4. *optimal*: 如果它的效果是正的就采取治疗。

        :returns: 第一个元素是下界，而第二个元素是上界。注意如果提供了协变量，所有的元素都是表明对应例子的上下界的维度为(n, )的ndarrays，其中n是例子的数量。
        :rtype: tuple
        :raises Exception: 如果模型没有拟合或者 :py:attr:`assump` 给的不正确，Raise Exception。

    .. py:method:: comp_transormer(x, categories='auto')
        
        把离散的治疗正确的转变为独热向量。

        :param numpy.ndarray, shape (n, x_d) x:  一个包含治疗变量信息的数组。
        :param str or list, optional, default='auto' categories:

        :returns: 转变后的独热变量。
        :rtype: numpy.ndarray
