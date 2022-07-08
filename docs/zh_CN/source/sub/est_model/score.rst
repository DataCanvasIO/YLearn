********************************
给估计的因果效应打分
********************************

用于估计因果效应的估计器模型不能被简单的评价，因为事实上真实的效应不能被直接观察到。这和平常的，结果可以用比如说损失函数的值简单评价的机器学习任务不同。

[Schuler]_ 的作者提出了一个框架，一个由 [Nie]_ 表明的模式，来评估不同的估计器模型估计的因果效应。粗略的说，这个框架是双机器学习方法的直接应用。
具体的说，对于一个因果效应模型 :py:func:`ce_model` （在训练集上训练好的），等待被评价，我们
    
1. 在通常和训练集不同的验证集中，训练一个模型 :py:func:`y_model` 来估计结果 :math:`y` 和一个 :py:func:`x_model` 来估计治疗 :math:`x` ；
2. 在验证集 :math:`D_{val}` 中，令 :math:`\tilde{y}` 和 :math:`\tilde{x}` 表示差
    
    .. math::

        \tilde{y} & = y - \hat{y}(v), \\
        \tilde{x} & = x - \hat{x}(v)
    
   其中 :math:`\hat{y}` 和 :math:`\hat{x}` 是在 :math:`D_{val}` 中估计的结果和治疗基于协变量 :math:`v` 。
   此外，令
    
    .. math::

        \tau(v)
    
   表明在 :math:`D_{val}` 中由 :py:func:`ce_model` 估计的因果效应，那么对于ce_model因果效应的度量标准这样计算。

    .. math::

        E_{V}[(\tilde{y} - \tilde{x} \tau(v))^2].

.. topic:: 例子

    .. code-block:: python

        from sklearn.ensemble import RandomForestRegressor

        from ylearn.exp_dataset.exp_data import single_binary_treatment
        from ylearn.estimator_model.meta_learner import TLearner
        
        train, val, te = single_binary_treatment()
        rloss = RLoss(
            x_model=RandomForestClassifier(),
            y_model=RandomForestRegressor(),
            cf_fold=1,
            is_discrete_treatment=True
        )
        rloss.fit(
            data=val,
            outcome=outcome,
            treatment=treatment,
            adjustment=adjustment,
            covariate=covariate,
        )
        
        est = TLearner(model=RandomForestRegressor())
        est.fit(
            data=train,
            treatment=treatment,
            outcome=outcome,
            adjustment=adjustment,
            covariate=covariate,
        )
    
    .. code-block:: python

        rloss.score(est)

    >>> 0.20451977

类结构
================

.. py:class:: ylearn.estimator_model.effect_score.RLoss(x_model, y_model, yx_model=None, cf_fold=1, adjustment_transformer=None, covariate_transformer=None, random_state=2022, is_discrete_treatment=False, categories='auto')

    :param estimator, optional x_model: 拟合x的机器学习模型。任何这样的模型应该实现  :py:func:`fit` 和 :py:func:`predict`` （也 :py:func:`predict_proba` 如果x是离散的）方法。
    :param estimator, optional y_model: 为了建模结果训练的机器学习模型。任何合理的y_model应该实现 :py:func:`fit()` 和 :py:func:`predict()` 方法。
    :param estimator, optional yx_model: 用于拟合基于x的残差的y的残差的机器学习模型。 *当前版本只支持线性回归模型。*
    
    :param int, default=1 cf_fold: 在第一阶段执行交叉拟合的折的数量。
    :param transormer, optional, default=None, adjustment_transformer: 调整变量的Transformer，其可以被用于生成调整变量的新特征。
    :param transormer, optional, default=None, covariate_transformer: 协变量的Transformer，其可以被用于生成协变量的新特征。
    :param int, default=2022 random_state:
    :param bool, default=False is_discrete_treatment: 如果治疗变量是离散的，把这个设为True。
    :param str, optional, default='auto' categories:
    
    .. py:method:: fit(data, outcome, treatment, adjustment=None, covariate=None, combined_treatment=True, **kwargs)
        
        拟合RLoss估计器模型。注意训练一个DML有两个阶段，其中我们在 :py:func:`_fit_1st_stage` 和 :py:func:`_fit_2nd_stage` 中实现它们。

        :param pandas.DataFrame data: 训练估计器的训练数据集。
        :param list of str, optional outcome: 结果的名字。
        :param list of str, optional treatment: 治疗的名字。
        :param list of str, optional, default=None adjustment: 保证无混淆的调整集的名字。
        :param list of str, optional, default=None covariate: 协变量的名字。
        :param bool, default=True combined_treatment: 当combined_treatment被设置为True时，那么如果有多个治疗，我们能使用combined_treatment技术
            来转变多个离散分类任务成为一个离散分类任务，比如，如果有两个不同的二元治疗：
            
            .. math::

                treatment_1 &: x_1 | x_1 \in \{'sleep', 'run'\}, \\
                treatment_2 &: x_2 | x_2 \in \{'study', 'work'\},
            
            那么我们能够转变这两个二元分类任务成为一个有四个不同类的分类任务。
                
            .. math::

                treatment: x | x \in \{0, 1, 2, 3\},
            
            其中，比如，1表示('sleep' and 'study')。

        :returns: RLoss的实例
        :rtype: 拟合的RLoss模型用于在验证集中评价其他的估计模型。

    .. py:method:: score(test_estimator, treat=None, control=None)
        
        用量的类型估计因果效应。

        :param pandas.DataFrame, optional, default=None data: 用于估计器估计因果效应的测试数据，注意如果data是None，估计器直接估计训练数据中所有的量。
        :param float or numpy.ndarray, optional, default=None treat: 在单个离散治疗的情况下，treat应该是所有可能的治疗值之一的int或者str，
            其表示预期的治疗值，在有多个离散治疗的情况下，treat应该是一个列表或者ndarray，其中treat[i]表示第i个预期的治疗值。例如，
            当有多个离散治疗，array(['run', 'read'])意味着第一个治疗的治疗值是 'run' ，第二个治疗是 'read' 。在连续治疗值的情况下，治疗应该是一个float或者ndarray。
        :param float or numpy.ndarray, optional, default=None control: 这和treat的情况相似。

        :returns: test_estimator的分数
        :rtype: float

    .. py:method:: effect_nji(data=None)
        
        用不同的治疗值计算因果效应。
        
        :param pandas.DataFrame, optional, default=None data: 用于估计器估计因果效应的测试数据，注意如果data是None，估计器会使用训练数据。

        :returns: 不同治疗值的因果效应。
        :rtype: ndarray

    .. py:method:: comp_transormer(x, categories='auto')
        
        把离散的治疗正确转变为独热向量。

        :param numpy.ndarray, shape (n, x_d) x:  包含治疗变量信息的数组。
        :param str or list, optional, default='auto' categories:

        :returns: 转变的独热向量。
        :rtype: numpy.ndarray

