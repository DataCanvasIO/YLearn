********************************
Scoring Estimated Causal Effects
********************************

Estimator models for estimating causal effects can not be easily evaluated
due to the fact that the true effects are not directly observed. This differs
from the usual machine learning tasks whoes results can be easily evaluated by, for example, the value of loss functions.

Authors in [Schuler]_ proposed a framework, a schema suggested by [Nie]_, to evaluate causal
effects estimated by different estimator models. Roughly speaking, this
framework is a direct application of the double machine learning methods.
Specifically, for a causal effect model :py:func:`ce_model` (trained in a training set)
that is waited to be evaluated, we 
    
1. Train a model :py:func:`y_model` to estimate the outcome :math:`y` and a :py:func:`x_model` to
   estimate the treatment :math:`x` in a validation set, which is usually not the same as the training set;
2. In the validation set :math:`D_{val}`, let :math:`\tilde{y}` and :math:`\tilde{x}` denote the differences
    
    .. math::

        \tilde{y} & = y - \hat{y}(v), \\
        \tilde{x} & = x - \hat{x}(v)
    
   where :math:`\hat{y}` and :math:`\hat{x}` are the estimated outcome and treatment on covariates :math:`v` in :math:`D_{val}`.
   Furthermore, let
    
    .. math::

        \tau(v)
    
   denote  the causal effects estimated by the :py:func:`ce_model` in :math:`D_{val}`, then the metric of the causal effect for the ce_model is
   calculated as

    .. math::

        E_{V}[(\tilde{y} - \tilde{x} \tau(v))^2].

.. topic:: Example

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

Class Structures
================

.. py:class:: ylearn.estimator_model.effect_score.RLoss(x_model, y_model, yx_model=None, cf_fold=1, adjustment_transformer=None, covariate_transformer=None, random_state=2022, is_discrete_treatment=False, categories='auto')

    :param estimator, optional x_model: Machine learning models for fitting x. Any such models should implement
            the :py:func:`fit` and :py:func:`predict`` (also :py:func:`predict_proba` if x is discrete) methods.
    :param estimator, optional y_model: The machine learning model which is trained to modeling the outcome. Any valid y_model should implement the :py:func:`fit()` and :py:func:`predict()` methods.
    :param estimator, optional yx_model: Machine learning models for fitting the residual of y on residual of x. *Only support linear regression model in the current version.*
    
    :param int, default=1 cf_fold: The nubmer of folds for performing cross fit in the first stage.
    :param transormer, optional, default=None, adjustment_transformer: Transformer for adjustment variables which can be used to generate new features of adjustment variables.
    :param transormer, optional, default=None, covariate_transformer: Transformer for covariate variables which can be used to generate new features of covariate variables.
    :param int, default=2022 random_state:
    :param bool, default=False is_discrete_treatment: If the treatment variables are discrete, set this to True.
    :param str, optional, default='auto' categories:
    
    .. py:method:: fit(data, outcome, treatment, adjustment=None, covariate=None, combined_treatment=True, **kwargs)
        
        Fit the RLoss estimator model. Note that the trainig of a DML has two stages, where we implement them in 
        :py:func:`_fit_1st_stage` and :py:func:`_fit_2nd_stage`.

        :param pandas.DataFrame data: Training dataset for training the estimator.
        :param list of str, optional outcome: Names of the outcome.
        :param list of str, optional treatment: Names of the treatment.
        :param list of str, optional, default=None adjustment: Names of the adjustment set ensuring the unconfoundness,
        :param list of str, optional, default=None covariate: Names of the covariate.
        :param bool, default=True combined_treatment: When combined_treatment is set to True, then if there are multiple
            treatments, we can use the combined_treatment technique to covert
            the multiple discrete classification tasks into a single discrete
            classification task. For an example, if there are two different
            binary treatments:
            
            .. math::

                treatment_1 &: x_1 | x_1 \in \{'sleep', 'run'\}, \\
                treatment_2 &: x_2 | x_2 \in \{'study', 'work'\},
            
            then we can convert to these two binary classification tasks into a single classification with 4 different classes:
                
            .. math::

                treatment: x | x \in \{0, 1, 2, 3\},
            
            where, for example, 1 stands for ('sleep' and 'stuy').

        :returns: instance of RLoss
        :rtype: The fitted RLoss model for evaluating other estimator models in the validation set.

    .. py:method:: score(test_estimator, treat=None, control=None)
        
        Estimate the causal effect with the type of the quantity.

        :param pandas.DataFrame, optional, default=None data: The test data for the estimator to evaluate the causal effect, note
            that the estimator directly evaluate all quantities in the training
            data if data is None.
        :param float or numpy.ndarray, optional, default=None treat: In the case of single discrete treatment, treat should be an int or
            str of one of all possible treatment values which indicates the
            value of the intended treatment;
            in the case of multiple discrete treatment, treat should be a list
            or an ndarray where treat[i] indicates the value of the i-th intended
            treatment, for example, when there are multiple
            discrete treatments, array(['run', 'read']) means the treat value of
            the first treatment is taken as 'run' and that of the second treatment
            is taken as 'read';
            in the case of continuous treatment, treat should be a float or a
            ndarray.
        :param float or numpy.ndarray, optional, default=None control: This is similar to the cases of treat.

        :returns: The score for the test_estimator
        :rtype: float

    .. py:method:: effect_nji(data=None)
        
        Calculate causal effects with different treatment values. 
        
        :param pandas.DataFrame, optional, default=None data: The test data for the estimator to evaluate the causal effect, note
            that the estimator will use the training data if data is None.

        :returns: Causal effects with different treatment values.
        :rtype: ndarray

    .. py:method:: comp_transormer(x, categories='auto')
        
        Transform the discrete treatment into one-hot vectors properly.

        :param numpy.ndarray, shape (n, x_d) x:  An array containing the information of the treatment variables.
        :param str or list, optional, default='auto' categories:

        :returns: The transformed one-hot vectors.
        :rtype: numpy.ndarray

