************
元学习器
************
元学习器是一种估计模型，旨在当处理手段为离散变量时通过机器学习模型去评估CATE。治疗方案为离散变量的意思也就是当无混淆条件下非1即0。通常来讲，它利用多个可灵活选择的机器学习模型。


YLearn 实现了3个元学习器: S-Learner, T-Learner, and X-Learner.

.. topic:: 示例

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

    **SLearner**

    .. code-block:: python

        s = SLearner(model=GradientBoostingRegressor())
        s.fit(data=data, outcome=outcome, treatment=treatment, adjustment=adjustment) # training
        s_pred = s.estimate(data=test_data, quantity=None) # predicting

    **TLearner**

    .. code-block:: python

        t = TLearner(model=GradientBoostingRegressor())
        t.fit(data=data, outcome=outcome, treatment=treatment, adjustment=adjustment) # training
        t_pred = t.estimate(data=test_data, quantity=None) # predicting

    **XLearner**

    .. code-block:: python

        x = XLearner(model=GradientBoostingRegressor())
        x.fit(data=data, outcome=outcome, treatment=treatment, adjustment=adjustment) # training
        x_pred = x.estimate(data=test_data, quantity=None) # predicting

S-Learner
=========

SLearner 采用一个机器学习模型来评估因果效应。具体来说，我们用机器学习模型 :math:`f` 从治疗方案 :math:`x` 和调整集 (或者协变量) :math:`w` 中拟合一个模型去预测结果 :math:`y`:

.. math::

    y = f(x, w).

因果效应 :math:`\tau(w)` 被计算为:

.. math::

    \tau(w) = f(x=1, w) - f(x=0, w).


.. py:class:: ylearn.estimator_model.meta_learner.SLearner(model, random_state=2022, is_discrete_treatment=True, categories='auto', *args, **kwargs)

    :param estimator, optional model: The base machine learning model for training SLearner. Any model
            should be some valid machine learning model with fit() and
            predict_proba() functions.
    :param int, default=2022 random_state:
    :param bool, default=True is_discrete_treatment: Treatment must be discrete for SLearner.
    :param str, optional, default='auto' categories:
    
    .. py:method:: fit(data, outcome, treatment, adjustment=None, covariate=None, treat=None, control=None, combined_treatment=True, **kwargs)
        
        Fit the SLearner in the dataset.

        :param pandas.DataFrame data: Training dataset for training the estimator.
        :param list of str, optional outcome: Names of the outcome.
        :param list of str, optional treatment: Names of the treatment.
        :param list of str, optional, default=None adjustment: Names of the adjustment set ensuring the unconfoundness,
        :param list of str, optional, default=None covariate: Names of the covariate.
        :param int, optional treat: Label of the intended treatment group
        :param int, optional control: Label of the intended control group
        :param bool, optional, default=True combined_treatment: Only modify this parameter for multiple treatments, where multiple discrete
                treatments are combined to give a single new group of discrete treatment if
                set as True. When combined_treatment is set to True, then if there are multiple
                treatments, we can use the combined_treatment technique to covert
                the multiple discrete classification tasks into a single discrete
                classification task. For an example, if there are two different
                binary treatments:
                    
                    1. treatment_1: :math:`x_1 | x_1 \in \{'sleep', 'run'\}`,
                    
                    2. treatment_2: :math:`x_2 | x_2 \in \{'study', 'work'\}`,
                
                then we can convert these two binary classification tasks into
                a single classification task with 4 different classes:
                
                treatment: :math:`x | x \in \{0, 1, 2, 3\}`,
                
                where, for example, 1 stands for ('sleep' and 'stuy').

        :returns: The fitted instance of SLearner.
        :rtype: instance of SLearner

    .. py:method:: estimate(data=None, quantity=None)
        
        Estimate the causal effect with the type of the quantity.

        :param pandas.DataFrame, optional, default=None data: Test data. The model will use the training data if set as None.
        :param str, optional, default=None quantity: Option for returned estimation result. The possible values of quantity include:
                
                1. *'CATE'* : the estimator will evaluate the CATE;
                
                2. *'ATE'* : the estimator will evaluate the ATE;
                
                3. *None* : the estimator will evaluate the ITE or CITE.

        :returns: The estimated causal effects 
        :rtype: ndarray

    .. py:method:: effect_nji(data=None)
        
        Calculate causal effects with different treatment values.

        :returns: Causal effects with different treatment values.
        :rtype: ndarray

    .. py:method:: _comp_transormer(x, categories='auto')
        
        Transform the discrete treatment into one-hot vectors properly.

        :param numpy.ndarray, shape (n, x_d) x:  An array containing the information of the treatment variables.
        :param str or list, optional, default='auto' categories:

        :returns: The transformed one-hot vectors.
        :rtype: numpy.ndarray

T-Learner
=========

TLearner的问题是当调整集向量为多维时治疗方案向量仅为一维。因此，如果调整集的维度超过1，那么评估结果将总是逼近于0。
TLearner用两个机器学习模型去评估因果效应。具体来讲，令 :math:`w` 为调整集（或协变量），我们

1. 分别拟合两个模型 :math:`f_t(w)` 对于治疗组 (:math:`x=` treat) 和 :math:`f_0(w)` 对于控制组 (:math:`x=` control):

    .. math::

        y_t = f_t(w)

  其中, :math:`x=` treat.

    .. math:: 

        y_0 = f_0(w)
    
  其中, :math:`x=` control.


2. 计算因果效应 :math:`\tau(w)` 作为两个模型预测结果的差异:

    .. math::

        \tau(w) = f_t(w) - f_0(w).

.. py:class:: ylearn.estimator_model.meta_learner.TLearner(model, random_state=2022, is_discrete_treatment=True, categories='auto', *args, **kwargs)

    :param estimator, optional model: The base machine learning model for training TLearner. Any model
            should be some valid machine learning model with fit() and
            predict_proba() functions.
    :param int, default=2022 random_state:
    :param bool, default=True is_discrete_treatment: Treatment must be discrete for SLearner.
    :param str, optional, default='auto' categories:
    
    .. py:method:: fit(data, outcome, treatment, adjustment=None, covariate=None, treat=None, control=None, combined_treatment=True, **kwargs)
        
        Fit the SLearner in the dataset.

        :param pandas.DataFrame data: Training dataset for training the estimator.
        :param list of str, optional outcome: Names of the outcome.
        :param list of str, optional treatment: Names of the treatment.
        :param list of str, optional, default=None adjustment: Names of the adjustment set ensuring the unconfoundness,
        :param list of str, optional, default=None covariate: Names of the covariate.
        :param int, optional treat: Label of the intended treatment group
        :param int, optional control: Label of the intended control group
        :param bool, optional, default=True combined_treatment: Only modify this parameter for multiple treatments, where multiple discrete
                treatments are combined to give a single new group of discrete treatment if
                set as True. When combined_treatment is set to True, then if there are multiple
                treatments, we can use the combined_treatment technique to covert
                the multiple discrete classification tasks into a single discrete
                classification task. For an example, if there are two different
                binary treatments:
                    
                    1. treatment_1: :math:`x_1 | x_1 \in \{'sleep', 'run'\}`,
                    
                    2. treatment_2: :math:`x_2 | x_2 \in \{'study', 'work'\}`,
                
                then we can convert these two binary classification tasks into
                a single classification task with 4 different classes:
                
                treatment: :math:`x | x \in \{0, 1, 2, 3\}`,
                
                where, for example, 1 stands for ('sleep' and 'stuy').

        :returns: The fitted instance of TLearner.
        :rtype: instance of TLearner

    .. py:method:: estimate(data=None, quantity=None)
        
        Estimate the causal effect with the type of the quantity.

        :param pandas.DataFrame, optional, default=None data: Test data. The model will use the training data if set as None.
        :param str, optional, default=None quantity: Option for returned estimation result. The possible values of quantity include:
                
                1. *'CATE'* : the estimator will evaluate the CATE;
                
                2. *'ATE'* : the estimator will evaluate the ATE;
                
                3. *None* : the estimator will evaluate the ITE or CITE.

        :returns: The estimated causal effects 
        :rtype: ndarray

    .. py:method:: effect_nji(data=None)
        
        Calculate causal effects with different treatment values.

        :returns: Causal effects with different treatment values.
        :rtype: ndarray

    .. py:method:: _comp_transormer(x, categories='auto')
        
        Transform the discrete treatment into one-hot vectors properly.

        :param numpy.ndarray, shape (n, x_d) x:  An array containing the information of the treatment variables.
        :param str or list, optional, default='auto' categories:

        :returns: The transformed one-hot vectors.
        :rtype: numpy.ndarray

X-Learner
=========

TLearner未能完全有效地利用数据，XLearner可以解决这个问题。训练一个XLearner可以分为3步:

1. 与TLearner类似, 我们首先分别训练两个不同的模型对于控制组和治疗组:

    .. math::

        & f_0(w) \text{for the control group}\\
        & f_1(w) \text{for the treat group}.

2. 生成两个新数据集 :math:`\{(h_0, w)\}` 用控制组, :math:`\{(h_1, w)\}`用治疗组。 其中
    
    .. math::

        h_0 & = f_1(w) - y_0,\\ 
        h_1 & = y_1 - f_0(w). 
    
   然后，训练两个机器学习模型在这些数据集中 :math:`k_0(w)` 和 :math:`k_1(w)`

    .. math::

        h_0 & = k_0(w) \\
        h_1 & = k_1(w).

3. 结合以上两个模型得到最终的模型:

    .. math::

        g(w) = k_0(w)a(w) + k_1(w)(1 - a(w))

    其中, :math:`a(w)` 是一个调整 :math:`k_0` 和 :math:`k_1` 的权重调整系数。

最后,  因果效应 :math:`\tau(w)` 通过以下方式评估:

.. math::

    \tau(w) = g(w).

.. py:class:: ylearn.estimator_model.meta_learner.XLearner(model, random_state=2022, is_discrete_treatment=True, categories='auto', *args, **kwargs)

    :param estimator, optional model: The base machine learning model for training SLearner. Any model
            should be some valid machine learning model with fit() and
            predict_proba() functions.
    :param int, default=2022 random_state:
    :param bool, default=True is_discrete_treatment: Treatment must be discrete for SLearner.
    :param str, optional, default='auto' categories:
    
    .. py:method:: fit(data, outcome, treatment, adjustment=None, covariate=None, treat=None, control=None, combined_treatment=True, **kwargs)
        
        Fit the SLearner in the dataset.

        :param pandas.DataFrame data: Training dataset for training the estimator.
        :param list of str, optional outcome: Names of the outcome.
        :param list of str, optional treatment: Names of the treatment.
        :param list of str, optional, default=None adjustment: Names of the adjustment set ensuring the unconfoundness,
        :param list of str, optional, default=None covariate: Names of the covariate.
        :param int, optional treat: Label of the intended treatment group
        :param int, optional control: Label of the intended control group
        :param bool, optional, default=True combined_treatment: Only modify this parameter for multiple treatments, where multiple discrete
                treatments are combined to give a single new group of discrete treatment if
                set as True. When combined_treatment is set to True, then if there are multiple
                treatments, we can use the combined_treatment technique to covert
                the multiple discrete classification tasks into a single discrete
                classification task. For an example, if there are two different
                binary treatments:
                    
                    1. treatment_1: :math:`x_1 | x_1 \in \{'sleep', 'run'\}`,
                    
                    2. treatment_2: :math:`x_2 | x_2 \in \{'study', 'work'\}`,
                
                then we can convert these two binary classification tasks into
                a single classification task with 4 different classes:
                
                treatment: :math:`x | x \in \{0, 1, 2, 3\}`,
                
                where, for example, 1 stands for ('sleep' and 'stuy').

        :returns: The fitted instance of XLearner.
        :rtype: instance of XLearner

    .. py:method:: estimate(data=None, quantity=None)
        
        Estimate the causal effect with the type of the quantity.

        :param pandas.DataFrame, optional, default=None data: Test data. The model will use the training data if set as None.
        :param str, optional, default=None quantity: Option for returned estimation result. The possible values of quantity include:
                
                1. *'CATE'* : the estimator will evaluate the CATE;
                
                2. *'ATE'* : the estimator will evaluate the ATE;
                
                3. *None* : the estimator will evaluate the ITE or CITE.

        :returns: The estimated causal effects 
        :rtype: ndarray

    .. py:method:: effect_nji(data=None)
        
        Calculate causal effects with different treatment values.

        :returns: Causal effects with different treatment values.
        :rtype: ndarray

    .. py:method:: _comp_transormer(x, categories='auto')
        
        Transform the discrete treatment into one-hot vectors properly.

        :param numpy.ndarray, shape (n, x_d) x:  An array containing the information of the treatment variables.
        :param str or list, optional, default='auto' categories:

        :returns: The transformed one-hot vectors.
        :rtype: numpy.ndarray
