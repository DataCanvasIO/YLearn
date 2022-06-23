************
Meta-Learner
************

Meta-Learners [Kunzel2019]_ are estimator models that aim to estimate the CATE by taking advantage of machine learning
models when the treatment is discrete, e.g., the treatment has only two values 1 and 0. Generally speaking,
it employs multiple machine learning models with the flexibility on the choice of models.

YLearn implements 3 Meta-Learners: S-Learner, T-Learner, and X-Learner.

.. topic:: Example

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

SLearner uses one machine learning model to estimate the causal effects. Specifically, we fit a model to predict outcome
:math:`y` from treatment :math:`x` and adjustment set (or covariate) :math:`w` with a machine learning model
:math:`f`:

.. math::

    y = f(x, w).

The causal effect :math:`\tau(w)` is then calculated as

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

The problem of SLearner is that the treatment vector is only 1-dimensional while the adjustment vector could be 
multi-dimensional thus if the dimension of the adjustment is much larger than 1 then the estimated result will always close to 0. 
TLearner uses two machine learning models to estimate the causal effect. Specifically, let :math:`w` denote the
adjustment set (or covariate), we

1. Fit two models :math:`f_t(w)` for the treatment group (:math:`x=` treat) and :math:`f_0(w)` for the control group (:math:`x=` control), respectively:

    .. math::

        y_t = f_t(w)

  with data where :math:`x=` treat and

    .. math:: 

        y_0 = f_0(w)
    
  with data where :math:`x=` control.


2. Compute the causal effect :math:`\tau(w)` as the difference between predicted results of these two models:

    .. math::

        \tau(w) = f_t(w) - f_0(w).

.. py:class:: ylearn.estimator_model.meta_learner.TLearner(model, random_state=2022, is_discrete_treatment=True, categories='auto', *args, **kwargs)

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

TLearnr does not use all data efficiently, which can be addressed by the XLearner. Training a XLearner is composed of 3 steps:

1. As in the case of TLearner, we first train two different models for the control group and treated group,  respectively:

    .. math::

        & f_0(w) \text{for the control group}\\
        & f_1(w) \text{for the treat group}.

2. Generate two new datasets :math:`\{(h_0, w)\}` using the control group and :math:`\{(h_1, w)\}` using the treated group where
    
    .. math::

        h_0 & = f_1(w) - y_0(w),\\ 
        h_1 & = y_1(w) - f_0(w). 
    
    Then train two new machine learing models :math:`k_0(w)` and :math:`k_1(w)` in these datasets such that

    .. math::

        h_0 & = k_0(w) \\
        h_1 & = k_1(w).

3. Get the final model by combining the above two models:

    .. math::

        g(w) = k_0(w)a(w) + k_1(w)(1 - a(w))

    where :math:`a(w)` is a coefficient adjusting the weight of :math:`k_0` and :math:`k_1`.

Finally,  the casual effect :math:`\tau(w)` can be estimated as follows:

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