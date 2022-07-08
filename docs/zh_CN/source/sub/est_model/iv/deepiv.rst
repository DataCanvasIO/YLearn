******
DeepIV
******

DeepIV，开发于 [Hartford]_ ，是一个在存在未观察到的混淆因素的情况下，估计治疗和结果变量之间因果效应的方法。当工具变量（IV）存在时，它应用深度
学习方法来准确表征治疗和结果之间的因果关系。由于深度学习模型的表示力，它没有假设因果关系的任何参数形式。

训练一个DeepIV有两步，类似于正常IV方法的估计过程。具体的，我们

    1. 训练一个神经网络，我们将其称为 *治疗网络* :math:`F(Z, V)` ，来估计治疗 :math:`X` 的分布，给定IV :math:`Z` 和协变量 :math:`V` 。

    2. 训练另一个神经网络，我们将其称为 *结果网络* :math:`H(X, V)` ，来估计结果 :math:`Y` 给定治疗 :math:`X` 和协变量 :math:`V`。

最终的因果效应接着可以被结果网络 :math:`H(X, W)` 估计。举个例子，CATE :math:`\tau(v)` 被这样估计

.. math::

    \tau(v) = H(X=x_t, V=v) - H(X=x_0, W=v).


类结构
================

.. py:class:: ylearn.estimator_model.deepiv.DeepIV(x_net=None, y_net=None, x_hidden_d=None, y_hidden_d=None, num_gaussian=5, is_discrete_treatment=False, is_discrete_outcome=False, is_discrete_instrument=False, categories='auto', random_state=2022)

    :param ylearn.estimator_model.deepiv.Net, optional, default=None x_net: 表示对于连续的治疗的混合密度网络或者是对于离散的治疗的常见的分类网络。如果是
            None，默认的神经网络将被使用。参考 :py:class:`ylearn.estimator_model.deepiv.Net` 。
    :param ylearn.estimator_model.deepiv.Net, optional, default=None y_net: 表示结果网络。如果是None，默认的神经网络将被使用。
    :param int, optional, default=None x_hidden_d: DeepIV默认的x_net的隐藏层的维度。
    :param int, optional, default=None y_hidden_d: DeepIV默认的y_net的隐藏层的维度。
    :param bool, default=False is_discrete_treatment:
    :param bool, default=False is_discrete_instrument:
    :param bool, default=False is_discrete_outcome:

    :param int, default=5 num_gaussian: 使用混合密度网络时的高斯数，当治疗是离散的时候，其将被直接忽略。
    :param int, default=2022 random_state:
    :param str, optional, default='auto' categories:
    
    .. py:method:: fit(data, outcome, treatment, instrument=None, adjustment=None, approx_grad=True, sample_n=None, y_net_config=None, x_net_config=None, **kwargs)
        
        训练DeepIV模型。

        :param pandas.DataFrame data: 训练估计器的训练数据集。
        :param list of str, optional outcome: 结果的名字。
        :param list of str, optional treatment: 治疗的名字。
        :param list of str, optional instrument: IV的名字。DeepIV必须提供。
        :param list of str, optional, default=None adjustment: 保证无混淆的调整集的名字，在当前版本其也可以被看作协变量。
        :param bool, default=True approx_grad:  是否使用近似梯度，和 [Hartford]_ 中一样。
        :param int, optional, default=None sample_n: 当使用approx_grad技术时，新样本的次数。
        :param dict, optional, default=None x_net_config: x_net的配置。
        :param dict, optional, default=None y_net_config: y_net的配置。
        
        :returns: 训练的DeepIV模型
        :rtype: DeepIV的实例

    .. py:method:: estimate(data=None, treat=None, control=None, quantity=None, marginal_effect=False, *args, **kwargs)
        
        用量的类型估计因果效应。

        :param pandas.DataFrame, optional, default=None data: 测试数据。注意被设置为None，模型会使用训练数据。
        :param str, optional, default=None quantity: 返回的估计结果的选项。量的可能值包括：
                
                1. *'CATE'* : 估计器将会估计CATE；
                
                2. *'ATE'* : 估计器将会估计ATE；
                
                3. *None* : 估计器将会估计ITE或CITE。
        :param int, optional, default=None treat: 治疗的值，默认是None。如果是None，那么模型会把treat设置为1。
        :param int, optional, default=None control: 控制的值，默认是None。如果是None，那么模型会把control设置为1。

        :returns: 估计的因果效应
        :rtype: torch.tensor

    .. py:method:: effect_nji(data=None)
        
        用不同的治疗值计算因果效应。

        :returns: 不同治疗值的因果效应。
        :rtype: ndarray

    .. py:method:: comp_transormer(x, categories='auto')
        
        把离散的治疗正确转变为独热向量。

        :param numpy.ndarray, shape (n, x_d) x:  包含治疗变量信息的数组。
        :param str or list, optional, default='auto' categories:

        :returns: 转变的独热向量。
        :rtype: numpy.ndarray