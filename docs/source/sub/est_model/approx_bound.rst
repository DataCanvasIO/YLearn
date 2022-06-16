*************************************
Approxmation Bound for Causal Effects
*************************************

Many estimator models require the unconfoundedness condition which is usually untestable. One applicable
approach is to build the upper and lower bounds of our causal effects before diving into specifical estimations.

There are four different bounds in YLearn. We briefly introduce them as follows. One can see [Neal2020]_ for details.

.. topic:: No-Assumptions Bound

    Suppose that
    
    .. math::

        \forall x, a \leq Y(do(x)) \leq b,
    
    then we have

    .. math::

        \mathbb{E}[Y(do(1)) - Y(do(0))] & \leq \pi \mathbb{E}[Y|X = 1] + (1 - \pi) b - \pi a - (1 - \pi )\mathbb{E}[Y| X = 0]\\
        \mathbb{E}[Y(do(1)) - Y(do(0))] & \geq \pi \mathbb{E}[Y|X = 1] + (1 - \pi) a - \pi b - (1 - \pi )\mathbb{E}[Y| X = 0]

    where :math:`\pi` is the probabiity of taking :math:`X=1`.

.. topic:: Nonnegative Monotone Treatment Response Bound

    Suppose that
    
    .. math::

        \forall i, Y(do(1)) \geq Y(do(0)),
    
    which means that *the treatment can only help*. Then we have the following bound:
    
    .. math::

        \mathbb{E}[Y(do(1)) - Y(do(0))] & \leq \pi \mathbb{E}[Y|X = 1] + (1 - \pi) b - \pi a - (1 - \pi )\mathbb{E}[Y| X = 0]\\
        \mathbb{E}[Y(do(1)) - Y(do(0))] & \geq 0

.. topic:: Nonpositive Monotone Treatment Response Bound

    Suppose that
    
    .. math::

        \forall i, Y(do(1)) \leq Y(do(0)),
    
    which means that *the treatment can never help*. Then we have the following bound:
    
    .. math::

        \mathbb{E}[Y(do(1)) - Y(do(0))] & \leq 0\\
        \mathbb{E}[Y(do(1)) - Y(do(0))] & \geq \pi \mathbb{E}[Y|X = 1] + (1 - \pi) a - \pi b - (1 - \pi )\mathbb{E}[Y| X = 0].
    
.. topic:: Optimal Treatment Selection Bound

    Suppose that
    
    .. math::
        
        X = 1 &\implies Y(do(1)) \geq Y(do(0)) \\
        X = 0 & \implies Y(do(0)) \geq Y(do(1))            

    which means that *people always receive the treatment if it is best for them*. Then we have the following bound:
    
    .. math::

        \mathbb{E}[Y(do(1)) - Y(do(0))] & \leq \pi \mathbb{E}[Y|X = 1] - \pi a\\
        \mathbb{E}[Y(do(1)) - Y(do(0))] & \geq (1 - \pi) a - (1 - \pi )\mathbb{E}[Y| X = 0].

    There are one more optimal treatment selection bound:

    .. math::

        \mathbb{E}[Y(do(1)) - Y(do(0))] & \leq \mathbb{E}[Y|X = 1] - \pi a - (1 - \pi)\mathbb{E}[Y|X=0]\\
        \mathbb{E}[Y(do(1)) - Y(do(0))] & \geq \pi\mathbb{E}[Y|X = 1] + (1 - \pi) a - \mathbb{E}[Y| X = 0].

Class Structures
================

.. py:class:: ylearn.estimator_model.approximation_bound.ApproxBound(y_model, x_prob=None, x_model=None, random_state=2022, is_discrete_treatment=True, categories='auto')

    A model used for estimating the upper and lower bounds of the causal effects.

    :param estimator, optional y_model: Any valide y_model should implement the fit() and predict() methods
    :param ndarray of shape (c, ), optional, default=None x_prob: An array of probabilities assigning to the corresponding values of x
            where c is the number of different treatment classes. All elements
            in the array are positive and sumed to 1. For example, x_prob = 
            array([0.5, 0.5]) means both x = 0 and x = 1 take probability 0.5.
            Please set this as None if you are using multiple treatmens.
    :param estimator, optional, default=None x_model: Models for predicting the probabilities of treatment. Any valide x_model should implement the fit() and predict_proba() methods.
    :param int, optional, default=2022 random_state:
    :param bool, optional, default=True is_discrete_treatment: True if the treatment is discrete.
    :param str, optional, default='auto' categories:

    .. py:method:: fit(data, outcome, treatment, covariate=None, is_discrete_covariate=False, **kwargs)
        
        Fit x_model and y_model.

        :param pandas.DataFrame data: Training data.
        :param list of str, optional outcome: Names of the outcome.
        :param list of str, optional treatment: Names of the treatment.
        :param list of str, optional, default=None covariate: Names of the covariate.
        :param bool, optional, default=False is_discrete_covariate:

        :returns: The fitted instance of ApproxBound.
        :rtype: instance of ApproxBound
        :raises ValueError:  Raise error when the treatment is not discrete.

    .. py:method:: estimate(data=None, treat=None, control=None, y_upper=None, y_lower=None, assump=None,)
        
        Estimate the approximation bound of the causal effect of the treatment
        on the outcome.

        :param pandas.DataFrame, optional, default=None data: Test data. The model will use the training data if set as None.
        :param ndarray of str, optional, default=None treat: Values of the treatment group. For example, when there are multiple
                discrete treatments, array(['run', 'read']) means the treat value of
                the first treatment is taken as 'run' and that of the second treatment
                is taken as 'read'. 
        :param ndarray of str, optional, default=None control: Values of the control group.
        :param float, defaults=None y_upper: The upper bound of the outcome.
        :param float, defaults=None y_lower: The lower bound of the outcome.
        :param str, optional, default='no-assump' assump: Options for the returned bounds. Should be one of
                
                1. *no-assump*: calculate the no assumption bound whose result will always contain 0.
                
                2. *non-negative*: The treatment is always positive.
                
                3. *non-positive*: The treatment is always negative.
                
                4. *optimal*: The treatment is taken if its effect is positive.

        :returns: The first element is the lower bound while the second element is the
                upper bound. Note that if covariate is provided, all elements are 
                ndarrays of shapes (n, ) indicating the lower and upper bounds of 
                corresponding examples where n is the number of examples. 
        :rtype: tuple
        :raises Exception: Raise Exception if the model is not fitted or if the :py:attr:`assump` is not given correctly.

    .. py:method:: comp_transormer(x, categories='auto')
        
        Transform the discrete treatment into one-hot vectors properly.

        :param numpy.ndarray, shape (n, x_d) x:  An array containing the information of the treatment variables.
        :param str or list, optional, default='auto' categories:

        :returns: The transformed one-hot vectors.
        :rtype: numpy.ndarray

.. topic:: Example

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