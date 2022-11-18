*************
Causal Forest
*************

In [Athey2018]_, the authors argued that by imposing the local centering technique, i.e., by first regressing out the effect and treatment respectively aka
the so called double machine learning framework, the performance of :ref:`grf` (GRF) can be further improved. In YLearn, we implement the class **CausalForest**
to support such technique. We illustrate its useage in the following example.

.. topic:: Example
    
    We first build a dataset and define the names of treatment, outcome, and covariate separately.

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt

        from ylearn.estimator_model import CausalForest
        from ylearn.exp_dataset.exp_data import sq_data
        from ylearn.utils._common import to_df


        # build dataset
        n = 2000
        d = 10     
        n_x = 1
        y, x, v = sq_data(n, d, n_x)
        true_te = lambda X: np.hstack([X[:, [0]]**2 + 1, np.ones((X.shape[0], n_x - 1))])
        data = to_df(treatment=x, outcome=y, v=v)
        outcome = 'outcome'
        treatment = 'treatment'
        adjustment = data.columns[2:]

        # build test data
        v_test = v[:min(100, n)].copy()
        v_test[:, 0] = np.linspace(np.percentile(v[:, 0], 1), np.percentile(v[:, 0], 99), min(100, n))
        test_data = to_df(v=v_test)
    
    Now it leaves us to train the `CausalForest` and use it in the test data. Typically, we should first specify two models which regressing out the treatment and outcome
    respectively on the covariate. In this example, we use the ``RandomForestRegressor`` from ``sklearn`` to be such models. Note that if we use a regression model for the
    treamtment, then the parameter ``is_discrete_treatment`` must be set as ``False``. To have better performance, it is also recommended to set the ``honest_subsample_num``
    as not ``None``.

    .. code-block:: python

        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LogisticRegressionCV

        cf = CausalForest(
            x_model=RandomForestRegressor(),
            y_model=RandomForestRegressor(),
            cf_fold=1,
            is_discrete_treatment=False,
            n_jobs=1,
            n_estimators=100,
            random_state=3,
            min_samples_split=10,
            min_samples_leaf=3,
            min_impurity_decrease=1e-10,
            max_depth=100,
            max_leaf_nodes=1000,
            sub_sample_num=0.80,
            verbose=0,
            honest_subsample_num=0.45,
        )
        cf.fit(data=data, outcome=outcome, treatment=treatment, adjustment=None, covariate=adjustment)
        effect = cf.estimate(test_data)

Class Structures
================


.. py:class:: ylearn.estimator_model.CausalForest(x_model, y_model, n_estimators=100, *, cf_fold=1, sub_sample_num=None, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0, n_jobs=None, random_state=None, ccp_alpha=0.0, is_discrete_treatment=True, is_discrete_outcome=False, verbose=0, warm_start=False, honest_subsample_num=None,)
   
    :param estimator, optional x_model: Machine learning models for fitting x. Any such models should implement
            the :py:func:`fit` and :py:func:`predict`` (also :py:func:`predict_proba` if x is discrete) methods.
   
    :param int, default=1 cf_fold: The number of folds for performing cross fit in the first stage.

    :param estimator, optional y_model: The machine learning model which is trained to modeling the outcome. Any valid y_model should implement the :py:func:`fit()` and :py:func:`predict()` methods.
   
    :param int, default=100 n_estimators: The number of trees for growing the GRF.

    :param int or float, default=None sub_sample_num: The number of samples to train each individual tree.
        - If a float is given, then the number of ``sub_sample_num*n_samples`` samples will be sampled to train a single tree
        - If an int is given, then the number of ``sub_sample_num`` samples will be sampled to train a single tree

    :param int, default=None max_depth: The max depth that a single tree can reach. If ``None`` is given, then there is no limit of
        the depth of a single tree.
    
    :param int, default=2 min_samples_split: The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    :param int or float, default=1 min_samples_leaf: The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.
            
            - If int, then consider `min_samples_leaf` as the minimum number.
            - If float, then `min_samples_leaf` is a fraction and `ceil(min_samples_leaf * n_samples)` are the minimum number of samples for each node.

    :param float, default=0.0 min_weight_fraction_leaf: The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
    
    :param int, float or {"sqrt", "log2"}, default=None max_features: The number of features to consider when looking for the best split:
        
            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and `int(max_features * n_features)` features are considered at each split.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

    :param int random_state: Controls the randomness of the estimator.
    
    :param int, default=None max_leaf_nodes: Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    :param float, default=0.0 min_impurity_decrease: A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
    
    :param int, default=None n_jobs: The number of jobs to run in parallel. :meth:`fit`, :meth:`estimate`, 
        and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors. See :term:`Glossary
        <n_jobs>` for more details.

    :param int, default=0 verbose: Controls the verbosity when fitting and predicting

    :param int or float, default=None honest_subsample_num: The number of samples to train each individual tree in an honest manner. Typically set this value will have better performance. Use all ``sub_sample_num`` if ``None`` is given.
        - If a float is given, then the number of ``honest_subsample_num*sub_sample_num`` samples will be used to train a single tree while the rest ``(1 - honest_subsample_num)*sub_sample_num`` samples will be used to label the trained tree.
        - If an int is given, then the number of ``honest_subsample_num`` samples will be sampled to train a single tree while the rest ``sub_sample_num - honest_subsample_num`` samples will be used to label the trained tree.

    .. py:method:: fit(data, outcome, treatment, adjustment=None, covariate=None)
        
        Fit the model on data to estimate the causal effect.

        :param pandas.DataFrame data: The input samples for the est_model to estimate the causal effects
            and for the CEInterpreter to fit.
        :param list of str, optional outcome: Names of the outcomes.
        :param list of str, optional treatment: Names of the treatments.
        :param list of str, optional, default=None covariate: Names of the covariate vectors.
        :param list of str, optional, default=None adjustment: This will be the same as the covariate.
        :param ndarray, optional, default=None sample_weight: Weight of each sample of the training set.
        
        :returns: Fitted GRForest
        :rtype: instance of GRForest

    .. py:method:: estimate(data=None)

        Estimate the causal effect of the treatment on the outcome in data.

        :param pandas.DataFrame, optional, default=None data: If None, data will be set as the training data.

        :returns: The estimated causal effect.
        :rtype: ndarray or float, optional


    .. .. py:method:: decision_path(*, data=None, wv=None)

    ..     Return the decision path.

    ..     :param numpy.ndarray, default=None wv: The input samples as an ndarray. If None, then the DataFrame data
    ..         will be used as the input samples.
    ..     :param pandas.DataFrame, default=None data: The input samples. The data must contains columns of the covariates
    ..         used for training the model. If None, the training data will be
    ..         passed as input samples.

    ..     :returns: Return a node indicator CSR matrix where non zero elements
    ..         indicates that the samples goes through the nodes.
    ..     :rtype: indicator : sparse matrix of shape (n_samples, n_nodes)

    .. py:method:: apply(*, v)

        Apply trees in the forest to v, return leaf indices.
        
        :param numpy.ndarray, v: The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``.

        :returns: For each datapoint v_i in v and for each tree in the forest,
            return the index of the leaf v ends up in.
        :rtype: v_leaves : array-like of shape (n_samples, )

    .. py:property:: feature_importance

        :returns: Normalized total reduction of criteria by feature (Gini importance).
        :rtype: ndarray of shape (n_features,)