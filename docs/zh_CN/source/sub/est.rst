.. _estimator_model:

**********************************************
估计器模型：估计因果效应
**********************************************

对于有 :math:`do`-operator 的因果效应，用被称为 :ref:`identification` 的方法把它转变为对应的统计估计量之后，因果推断的任务现在变成估计统计估计量，
即转变后的因果效应。在深入到任何具体的因果效应估计方法之前，我们简要的介绍因果效应估计的问题设置。

问题设置
===============

在 :ref:`causal_model` 中介绍了每个因果结构都有对应的被称为因果图的DAG。此外，一个DAG :math:`G` 的每一个子父家庭表示一个确定性函数。

.. math::

    X_i = F_i (pa_i, \eta_i), i = 1, \dots, n,

其中， :math:`pa_i` 是 :math:`x_i` 在 :math:`G` 中的父节点，且 :math:`\eta_i` 是随机扰动，表示外生的未在分析中出现的。我们称这些函数为
关于因果结构的 **Structural Equation Model** 。对于一组满足后门准则（参考 :ref:`identification`）的变量 :math:`W`，:math:`X` 对 :math:`Y` 的因果效应由公式给出

.. math::

    P(y|do(x))  = \sum_w P(y| x, w)P(w).

在这样的情况下，上述等式有效的变量 :math:`X` 也被命名为 *"条件可忽略的给定* :math:`W`" 在 *潜在结果* 的框架中。 满足这个条件的变量组 :math:`W` 被
称为 **调整集合** 。在结构化方程模型的语言里，这些关系编码如

.. math::

    X & = F_1 (W, \epsilon),\\
    Y & = F_2 (W, X, \eta).

我们的问题可以用结构化方程模型表示。

.. topic:: ATE

    具体来说，YLearn中一个尤其重要的因果量是差

    .. math::

        \mathbb{E}(Y|do(X=X_1)) - \mathbb{E}(Y|do(X=X_0))

    也被称为 **平均治疗效果 (ATE)**，其中 :math:`Y` 被称为 *结果* 而 :math:`X` 被称为 *治疗* 。此外，当条件独立（条件可忽略性）成立时，给定一组
    变量 :math:`W` 对结果 :math:`Y`` 和治疗 :math:`X` 都有潜在的影响，ATE能够这样估计

    .. math::

        E(Y|X=x_1, w) - E(Y|X=x_0, w).

    使用结构化方程模型，我们可以把上述关系描述为

    .. math::
        
        X & = F_1 (W, \epsilon) \\
        Y & = F_2 (X, W, \eta) \\
        \text{ATE} & = \mathbb{E}\left[ F_2(x_1, W, \eta) - F_2(x_0, W, \eta)\right]. 

.. topic:: CATE

    假如我们给调整集合 :math:`W`` 的一个子集的变量分配了特殊的角色并把它们命名为 **协变量** :math:`V` ，那么，在结构化方程模型
    中， **CATE** （也被称为 **异质治疗效果**）被定义为
    
    .. math::
    
        X & = F_1 (W, V, \epsilon) \\
        Y & = F_2 (X, W, V, \eta) \\
        \text{CATE} & = \mathbb{E}\left[ f_2(x_1, W, V, \eta) - f_2(x_0, W, V, \eta)| V =v\right].

.. topic:: 反事实

    除了是效应的差的因果估计量，还有一个因果量 **反事实** 。对于这个量，我们估计如下的因果估计量：

    .. math::

        \mathbb{E} [Y|do(x), V=v].


估计器模型
==========================
YLearn实现了几个用于估计因果效应的估计器模型

.. toctree::
    :maxdepth: 2

    est_model/approx_bound
    est_model/meta
    est_model/dml
    est_model/dr
    est_model/causal_tree
    est_model/iv
    est_model/score
    
对

.. math::
    
    \mathbb{E}[F_2(x_1, W, \eta) - F_2(x_0, W, \eta)]
    
在ATE中和

.. math::
    
    \mathbb{E}[F_2(x_1, W, V, \eta) - F_2(x_0, W, V, \eta)]

在CATE中的估计将会成为YLearn中不同的合适的  **估计器模型** 的任务。YLearn中概念 :class:`EstimatorModel` 就是为这个目的设计的

一个常见的 :class:`EstimatorModel` 应该有如下结构：

.. code-block:: python

    class BaseEstModel:
        """
        Base class for various estimator model.

        Parameters
        ----------
        random_state : int, default=2022
        is_discrete_treatment : bool, default=False
            Set this to True if the treatment is discrete.
        is_discrete_outcome : bool, default=False
            Set this to True if the outcome is discrete.            
        categories : str, optional, default='auto'

        """
        def fit(
            self,
            data,
            outcome,
            treatment,
            **kwargs,
        ):
            """Fit the estimator model.

            Parameters
            ----------
            data : pandas.DataFrame
                The dataset used for training the model

            outcome : str or list of str, optional
                Names of the outcome variables

            treatment : str or list of str
                Names of the treatment variables

            Returns
            -------
            instance of BaseEstModel
                The fitted estimator model.
            """

        def estimate(
            self,
            data=None,
            quantity=None,
            **kwargs
        ):
            """Estimate the causal effect.

            Parameters
            ----------
            data : pd.DataFrame, optional
                The test data for the estimator to evaluate the causal effect, note
                that the estimator directly evaluate all quantities in the training
                data if data is None, by default None

            quantity : str, optional
                The possible values of quantity include:
                    'CATE' : the estimator will evaluate the CATE;
                    'ATE' : the estimator will evaluate the ATE;
                    None : the estimator will evaluate the ITE or CITE, by default None
    
            Returns
            -------
            ndarray
                The estimated causal effect with the type of the quantity.
            """

        def effect_nji(self, data=None, *args, **kwargs):
            """Return causal effects for all possible values of treatments.

            Parameters
            ----------
            data : pd.DataFrame, optional
                The test data for the estimator to evaluate the causal effect, note
                that the estimator directly evaluate all quantities in the training
                data if data is None, by default None
            """

.. topic:: 用法

    可以在以下过程中应用任何 :class:`EstimatorModel`

    1. 对于 :class:`pandas.DataFrame` 形式的数据，找到 *治疗* ， *结果* ， *调整* ，和 *协变量* 的名字。

    2. 把数据和治疗，结果，调整集合，协变量的名字传入 :class:`EstimatorModel` 的 :meth:`fit()` 方法并调用它。

    3. 调用 :meth:`estimate()` 方法在测试数据上使用拟合的 :class:`EstimatorModel`。