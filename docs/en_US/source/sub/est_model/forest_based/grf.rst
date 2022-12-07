.. _grf:


*************************
Generalized Random Forest
*************************

To adpat random forest to causal effect estimation, [Athey2018]_ proposed a generalized version of it, named as Generalized Random Forest (GRF), by altering the criterion
when building a single tree and designing a new kind of ensemble method to combine these trained trees. GRF can be used in, for example, quantile regression while in YLearn,
we focus on its ability of performing highly flexible non-parametric causal effect estimation.

We now consider such estimation with GRF. Suppose that we observe samples :math:`(X_i, Y_i, V_i) \in \mathbb{R}^{d_x} \times \mathbb{R} \times \mathbb{R}^{d_v}` where :math:`Y`
is the outcome, :math:`X` is the treatment and :math:`V` is the covariate which ensures the unconfoundness condition. The forest weights :math:`\alpha_i(v)` is defined by

.. math::

    \alpha_i^b(v) = \frac{\mathbb{I}\left( \left\{ V_i \in L^b(v) \right\} \right)}{|L^b(v)|},\\
    \alpha_i(v) = \frac{1}{B} \sum_{b = 1}^B \alpha_i^b(v),

where the subscript :math:`b` refers to the :math:`b`-th tree with a total number of :math:`B` such trees, :math:`L^b(v)` is the leaf that the sample which covariate :math:`v`
belongs to, and :math:`|L^b(v)|` denotes the total number of training samples which fall into the samel leaf as the sample :math:`v` for the :math:`b`-th tree. Then the estimated
causal effect can be expressed by

.. math::

    \left( \sum_{i=1}^n \alpha_i(x)(X_i - \bar{X}_\alpha)(X_i - \bar{X}_\alpha)^T\right)^{-1} \sum_{i = 1}^n \alpha_i(v) (X_i - \bar{X}_\alpha)(Y_i - \bar{Y}_\alpha)

where :math:`\bar{X}_\alpha = \sum \alpha_i X_i` and :math:`\bar{Y}_\alpha = \sum \alpha_i Y_i`.

We now provide an example useage of applying the ``GRForest``.

.. topic:: Example
    
    We first build a dataset and define the names of treatment, outcome, and covariate separately.

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt

        from ylearn.estimator_model import GRForest
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
        covariate = data.columns[2:]

        # build test data
        v_test = v[:min(100, n)].copy()
        v_test[:, 0] = np.linspace(np.percentile(v[:, 0], 1), np.percentile(v[:, 0], 99), min(100, n))
        test_data = to_df(v=v_test)
    
    We now train the `GRForest` and use it in the test data. To have better performance, it is also recommended to set the ``honest_subsample_num``
    as not ``None``.

    .. code-block:: python

        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LogisticRegressionCV

        grf = GRForest(
            n_jobs=1, 
            honest_subsample_num=None,
            min_samples_split=10, 
            sub_sample_num=0.5, 
            n_estimators=100, 
            random_state=2022, 
            min_impurity_decrease=1e-10, 
            max_depth=100, 
            max_leaf_nodes=100, 
            verbose=0,
        )
        grf.fit(
            data=data, outcome=outcome, treatment=treatment, covariate=covariate
        )
        effect = grf.estimate(test_data)

Besides this ``GRForest``, YLearn also implements a naive version of GRF with pure python in an easy to understand manner to help users get some insights on how GRF works in code level.
It is worth to mention that, however, this naive version of GRF is super slow (~5mins for fitting 100 trees in a dataset with 2000 samples and 10 features). One can find this naive
GRF in the folder ylearn/estimator_model/_naive_forest/. 

The formal version of GRF is summarized as follows.

Class Structures
================

.. py:class:: ylearn.estimator_model.GRForest(n_estimators=100, *, sub_sample_num=None, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0, n_jobs=None, random_state=None, ccp_alpha=0.0, is_discrete_treatment=True, is_discrete_outcome=False, verbose=0, warm_start=False, honest_subsample_num=None,)

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

    :param int or float, default=None honest_subsample_num: The number of samples to train each individual tree in an honest manner. Typically setting this value will have better performance. 
        
        - Use all ``sub_sample_num`` if ``None`` is given.
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

        Apply trees in the forest to X, return leaf indices.
        
        :param numpy.ndarray, v: The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``.

        :returns: For each datapoint v_i in v and for each tree in the forest,
            return the index of the leaf v ends up in.
        :rtype: v_leaves : array-like of shape (n_samples, )

    .. py:property:: feature_importance

        :returns: Normalized total reduction of criteria by feature (Gini importance).
        :rtype: ndarray of shape (n_features,)