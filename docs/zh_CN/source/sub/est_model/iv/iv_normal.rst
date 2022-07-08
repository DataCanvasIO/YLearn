************************************
无参工具变量
************************************

两阶段最小平方
=======================
当结果 :math:`y` ，治疗 :math:`x` 和协变量 covariate :math:`v` 之间的关系假设是线性的，比如， [Angrist1996]_ ，

.. math::

    y & = \alpha x + \beta v + e \\
    x & = \gamma z + \lambda v + \eta,

那么IV框架变得直接：它首先给定 :math:`z` 和 :math:`v` 为 :math:`x` 训练一个线性模型。接着，在第二阶段，它把 :math:`x` 替换为估计值 :math:`\hat{x}` 来
为 :math:`y` 训练一个线性模型。这个过程被称为两阶段最小平方(2SLS)。

无参IV
================
移除关于变量之间关系的线性假设，无参IV能够取代线性回归，通过线性投影到一系列有名的基函数 [Newey2002]_ 。

这个方法和传统的2SLS类似且在找到 :math:`x` ， :math:`v` ，和 :math:`z` 的新特征之后也由两个阶段组成，

.. math:: 
    
    \tilde{z}_d & = f_d(z)\\
    \tilde{v}_{\mu} & = g_{\mu}(v),

其由一些非线性函数（基函数） :math:`f_d` 和 :math:`g_{\mu}` 表示。在变换到新的空间后，我们接着
    
    1. 拟合治疗模型：
    
    .. math::

        \hat{x}(z, v, w) = \sum_{d, \mu} A_{d, \mu} \tilde{z}_d \tilde{v}_{\mu} + h(v, w) + \eta
    
    2. 生成新的治疗x_hat，接着拟合结果模型

    .. math::
        
        y(\hat{x}, v, w) = \sum_{m, \mu} B_{m, \mu} \psi_m(\hat{x}) \tilde{v}_{\mu} + k(v, w) 
        + \epsilon.

最终因果效应能够被估计。举个例子，给定 :math:`v` ，CATE被估计为
    
    .. math::
        
        y(\hat{x_t}, v, w) - y(\hat{x_0}, v, w) = \sum_{m, \mu} B_{m, \mu} (\psi_m(\hat{x_t}) - \psi_m(\hat{x_0})) \tilde{v}_{\mu}.

YLearn在类 :class:`NP2SLS` 中实现了这个过程。

类结构
================

.. py:class:: ylearn.estimator_model.iv.NP2SLS(x_model=None, y_model=None, random_state=2022, is_discrete_treatment=False, is_discrete_outcome=False, categories='auto')

    :param estimator, optional, default=None x_model: 为了建模治疗的机器学习模型。任何合理的x_model应该实现 `fit` 和 `predict` 方法，默认是None。
    :param estimator, optional, default=None y_model: 为了建模结果的机器学习模型。任何合理的y_model应该实现 `fit` 和 `predict` 方法，默认是None。
    :param int, default=2022 random_state:
    :param bool, default=False is_discrete_treatment: 
    :param bool, default=False is_discrete_outcome: 
    :param str, optional, default='auto' categories:

    .. py:method:: fit(data, outcome, treatment, instrument, is_discrete_instrument=False, treatment_basis=('Poly', 2), instrument_basis=('Poly', 2), covar_basis=('Poly', 2), adjustment=None, covariate=None, **kwargs)

        拟合NP2SLS。注意当treatment_basis和instrument_basis都有degree 1的时候，我们实际在做2SLS。

        :param DataFrame data: 模型的训练数据集。Training data for the model.
        :param str or list of str, optional outcome: 结果的名字。
        :param str or list of str, optional treatment: 治疗的名字。
        :param str or list of str, optional, default=None covariate: 协变量向量的名字。
        :param str or list of str, optional instrument: 工具变量的名字。
        :param str or list of str, optional, default=None adjustment: 调整变量的名字。Names of the adjustment variables.
        :param tuple of 2 elements, optional, default=('Poly', 2) treatment_basis: 转换原来的治疗向量的选项。第一个元素表示转换的基函数，第二个表示degree。现在第一个元素只支持'Poly'。
        :param tuple of 2 elements, optional, default=('Poly', 2) instrument_basis: 转换原来的工具向量的选项。第一个元素表示转换的基函数，第二个表示degree。现在第一个元素只支持'Poly'。
        :param tuple of 2 elements, optional, default=('Poly', 2) covar_basis: 转换原来的协变量向量的选项。第一个元素表示转换的基函数，第二个表示degree。现在第一个元素只支持'Poly'。
        :param bool, default=False is_discrete_instrument:

    .. py:method:: estimate(data=None, treat=None, control=None, quantity=None)

        估计数据中治疗对结果的因果效应。

        :param pandas.DataFrame, optional, default=None data: 如果为None，数据将会被设置为训练数据。
        :param str, optional, default=None quantity: 返回的估计结果的选项。量的可能值包括：
                
                1. *'CATE'* : 估计器将会估计CATE；
                
                2. *'ATE'* : 估计器将会估计ATE；
                
                3. *None* : 估计器将会估计ITE或CITE。
        :param float, optional, default=None treat: 施加干涉时治疗的值。如果是None，那么treat将会被设置为1。
        :param float, optional, default=None control: 治疗的值，这样治疗的效果是 :math:`y(do(x=treat)) - y (do(x = control))` 。

        :returns: 用量的类型估计的因果效应。
        :rtype: ndarray or float, optional

    .. py:method:: effect_nji(data=None)

        用不同的治疗值计算因果效应。
        
        :param pandas.DataFrame, optional, default=None data: 给估计器估计因果效应的测试数据，注意如果data是None，估计器将会使用训练数据。

        :returns: 用不同的治疗值的因果效应。
        :rtype: ndarray

.. topic:: 例子

    pass
