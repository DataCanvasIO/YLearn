***********************
双机器学习
***********************

.. topic:: 符号

    我们使用大写字母表示矩阵，小写字母表示向量。治疗由 :math:`x` 表示，结果由 :math:`y` 表示，协变量由 :math:`v` 表示，且其他的调整集合变量是 :math:`w`。
    希腊字母用于错误项。

双机器学习（DML）模型 [Chern2016]_ 适用于当治疗，结果，变量的同时影响治疗和结果的所有的混杂因素都被观察到。令 :math:`y` 为结果，:math:`x` 为治疗，一个
DML模型解决如下的因果效应估计（CATE估计）：

.. math::

    y & = F(v) x + g(v, w) + \epsilon \\
    x & = h(v, w) + \eta

其中 :math:`F(v)` 是CATE以 :math:`v` 为条件。 此外，为了估计 :math:`F(v)`，我们注意到

.. math::

    y - \mathbb{E}[y|w, v] = F(v) (x - \mathbb{E}[x|w, v]) + \epsilon. 
    
因此通过首先估计 :math:`\mathbb{E}[y|w, v]` 和 :math:`\mathbb{E}[x|w,v]` 为

.. math::

    m(v, w) & = \mathbb{E}[y|w, v]\\
    h(v, w) & = \mathbb{E}[x|w,v],

我们能够得到一个新的数据集： :math:`(\tilde{y}, \tilde{x})` 其中

.. math::

    \tilde{y} & = y - m(v, w) \\
    \tilde{x} & = x - h(v, w)

这样 :math:`\tilde{y}` 和 :math:`\tilde{x}` 之间的关系是线性的

.. math::

    \tilde{y} = F(v) \tilde(x) + \epsilon

其能够被线性回归模型简单的建模。

另一方面，在现在的版本， :math:`F(v)` 采取形式

.. math::

    F_{ij}(v) = \sum_k H_{ijk} \rho_k(v).
    
其中 :math:`H` 能够被看作一个秩为3的张量且 :math:`\rho_k` 是协变量 :math:`v` 的函数，比如，最简单的情况 :math:`\rho(v) = v` 。因此，
结果 :math:`y` 现在能够被表示为

.. math::

    y_i & = \sum_j F_{ij}x_j + g(v, w)_j + \epsilon \\
        & = \sum_j \sum_k H_{ijk}\rho_k(v)x_j + g(v, w)_j + \epsilon

在这个意义上，:math:`\tilde{y}` 和 :math:`\tilde{x}` 之间的线性回归问题现在成为，

.. math::

    \tilde{y}_i = \sum_j \sum_k H_{ijk}\rho_k(v) \tilde{x}_j + \epsilon.

.. topic:: 实现

    在YLearn中，我们实现了一个双机器学习模型如 [Chern2016]_ 中描述的算法：

        1. 令 k (cf_folds in our class) 为一个int. 形成一个k-折随机划分{..., (train_data_i, test_data_i), ...,
        (train_data_k, test_data_k)}。

        2. 对于每个i，训练y_model和x_model在train_data_i上，接着评估它们的性能在test_data_i中，其结果会被保存为 :math:`(\hat{y}_k, \hat{x}_k)` 。
        所有的 :math:`(\hat{y}_k, \hat{x}_k)` 将会合并以提供新的数据集 :math:`(\hat{y}, \hat{x})` 。

        3. 定义差

        .. math::

            \tilde{y}& = y - \hat{y}, \\
            \tilde{x}&= (x - \hat{x}) \otimes v.

        接着形成新的数据集 :math:`(\tilde{y}, \tilde{x})`.

        4. 在数据集 :math:`(\tilde{y}, \tilde{x})` 上执行线性回归，其系数将会被保存到向量 :math:`f` 中。给定 :math:`v`，估计的CATE将会为

        .. math::

            f \cdot v.

.. topic:: 例子

    .. code-block:: python
        
        from sklearn.ensemble import RandomForestRegressor

        from ylearn.exp_dataset.exp_data import single_continuous_treatment
        from ylearn.estimator_model.double_ml import DoubleML

        # build the dataset
        train, val, treatment_effect = single_continuous_treatment()
        adjustment = train.columns[:-4]
        covariate = 'c_0'
        outcome = 'outcome'
        treatment = 'treatment'

        dml = DoubleML(x_model=RandomForestRegressor(), y_model=RandomForestRegressor(), cf_fold=3,)
        dml.fit(train, outcome, treatment, adjustment, covariate,)

    >>> 06-23 14:02:36 I ylearn.e.double_ml.py 684 - _fit_1st_stage: fitting x_model RandomForestRegressor
    >>> 06-23 14:02:39 I ylearn.e.double_ml.py 690 - _fit_1st_stage: fitting y_model RandomForestRegressor
    >>> DoubleML(x_model=RandomForestRegressor(), y_model=RandomForestRegressor(), yx_model=LinearRegression(), cf_fold=3)      

类结构
================

.. py:class:: ylearn.estimator_model.double_ml.DoubleML(x_model, y_model, yx_model=None, cf_fold=1, adjustment_transformer=None, covariate_transformer=None, random_state=2022, is_discrete_treatment=False, categories='auto')

    :param estimator, optional x_model: 拟合x的机器学习模型。任何这样的模型应该实现 :py:func:`fit` 和 :py:func:`predict`` （也 :py:func:`predict_proba` 如果x是离散的）方法。
    :param estimator, optional y_model: 为了建模结果训练的机器学习模型。任何合理的y_model应该实现 :py:func:`fit()` 和 :py:func:`predict()` 方法。
    :param estimator, optional yx_model: 用于拟合基于x的残差的y的残差的机器学习模型。 *当前版本只支持线性回归模型。*
    
    :param int, default=1 cf_fold: 在第一阶段执行交叉拟合的折的数量。
    :param transormer, optional, default=None, adjustment_transformer: 调整变量的Transformer，其可以被用于生成调整变量的新特征。
    :param transormer, optional, default=None, covariate_transformer: 协变量的Transformer，其可以被用于生成协变量的新特征。
    :param int, default=2022 random_state:
    :param bool, default=False is_discrete_treatment: 如果治疗变量是离散的，把这个设为True。
    :param str, optional, default='auto' categories:
    
    .. py:method:: fit(data, outcome, treatment, adjustment=None, covariate=None, **kwargs)
        
        拟合DoubleML估计器模型。注意训练一个DML有两个阶段，其中我们在 :py:func:`_fit_1st_stage` 和 :py:func:`_fit_2nd_stage` 中实现它们。

        :param pandas.DataFrame data: 训练估计器的训练数据集。
        :param list of str, optional outcome: 结果的名字。
        :param list of str, optional treatment: 治疗的名字。
        :param list of str, optional, default=None adjustment: 保证无混淆的调整集的名字。
        :param list of str, optional, default=None covariate: 协变量的名字。

        :returns: 拟合的model
        :rtype: 一个DoubleML的实例

    .. py:method:: estimate(data=None, treat=None, control=None, quantity=None)
        
        用量的类型估计因果效应。

        :param pandas.DataFrame, optional, default=None data: 用于估计器估计因果效应的测试数据，注意如果data是None，估计器直接估计训练数据中所有的量。
        :param float or numpy.ndarray, optional, default=None treat: 在单个离散治疗的情况下，treat应该是所有可能的治疗值之一的int或者str，
            其表示预期的治疗值，在有多个离散治疗的情况下，treat应该是一个列表或者ndarray，其中treat[i]表示第i个预期的治疗值。例如，
            当有多个离散治疗，array(['run', 'read'])意味着第一个治疗的治疗值是 'run' ，第二个治疗是 'read' 。在连续治疗值的情况下，治疗应该是一个float或者ndarray。
        :param str, optional, default=None quantity: 返回的估计结果的选项。量的可能值包括：
                
                1. *'CATE'* : 估计器将会估计CATE；
                
                2. *'ATE'* : 估计器将会估计ATE；
                
                3. *None* : 估计器将会估计ITE或CITE。
        :param float or numpy.ndarray, optional, default=None control: 这和treat的情况相似。

        :returns: 估计的因果效应
        :rtype: ndarray

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

