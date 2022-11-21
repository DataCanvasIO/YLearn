************************
Ensemble of Causal Trees
************************

An efficient and useful technique for growing a random forest is by simply averaging the result of each
individual tree. Consequently, we can also apply this technique to grow a causal forest by combining many single causal tree.
In YLearn, we implement this idea in the class :py:class:`CTCausalForest` (refering to Causal Tree Causal Forest).

Since it is an ensemble of a bunch of CausalTree, currently it only supports binary treatment. One may need specify the treat and control groups
before applying the CTCausalForest. This will be improved in the future version.

We provide below an example of it.

.. topic:: Example
    
    We first build a dataset and define the names of treatment, outcome, and covariate separately.

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt

        from ylearn.estimator_model import CTCausalForest
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
    
    We now train the `CTCausalForest` and use it in the test data. To have better performance, it is also recommended to set the ``honest_subsample_num``
    as not ``None``. 

    .. code-block:: python

        ctcf = CTCausalForest(
            n_jobs=-1, 
            honest_subsample_num=0.5,
            min_samples_split=2,
            min_samples_leaf=10, 
            sub_sample_num=0.8, 
            n_estimators=500, 
            random_state=2022, 
            min_impurity_decrease=1e-10, 
            max_depth=100, 
            max_features=0.8,
            verbose=0,
        )
        ctcf.fit(data=data, outcome=outcome, treatment=treatment, adjustment=adjustment)
        ctcf_pred = ctcf.estimate(data=test_data)

Class Structures
================

.. py:class:: ylearn.estimator_model.CTCausalForest(n_estimators=100, *, sub_sample_num=None, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=1.0, min_impurity_decrease=0.0, n_jobs=None, random_state=None, ccp_alpha=0.0, is_discrete_treatment=True, is_discrete_outcome=False, verbose=0, warm_start=False, honest_subsample_num=None,)

    :param int, default=100 n_estimators: The number of trees for growing the CTCausalForest.

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
    
    :param int, float or {"sqrt", "log2"}, default=None max_features: The number of features to consider when looking for the best split:
        
            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and `int(max_features * n_features)` features are considered at each split.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

    :param int random_state: Controls the randomness of the estimator.

    :param float, default=0.0 min_impurity_decrease: A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
    
    :param int, default=None n_jobs: The number of jobs to run in parallel. :meth:`fit`, :meth:`estimate`, 
        and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors.

    :param int, default=0 verbose: Controls the verbosity when fitting and predicting

    :param int or float, default=None honest_subsample_num: The number of samples to train each individual tree in an honest manner. Typically setting this value will have better performance. 
    
        - Use all ``sub_sample_num`` if ``None`` is given.
        - If a float is given, then the number of ``honest_subsample_num*sub_sample_num`` samples will be used to train a single tree while the rest ``(1 - honest_subsample_num)*sub_sample_num`` samples will be used to label the trained tree.
        - If an int is given, then the number of ``honest_subsample_num`` samples will be sampled to train a single tree while the rest ``sub_sample_num - honest_subsample_num`` samples will be used to label the trained tree.

    .. py:method:: fit(data, outcome, treatment, adjustment=None, covariate=None, treat=None, control=None)
        
        Fit the model on data to estimate the causal effect. Note that similar to CausalTree, currently CTCausalForest assumes a binary treatment where the values
        of ``treat`` and ``control`` are controled by the corresponding parameters.

        :param pandas.DataFrame data: The input samples for the est_model to estimate the causal effects
            and for the CEInterpreter to fit.
        :param list of str, optional outcome: Names of the outcomes.
        :param list of str, optional treatment: Names of the treatments.
        :param list of str, optional, default=None covariate: Names of the covariate vectors.
        :param list of str, optional, default=None adjustment: This will be the same as the covariate.
        :param ndarray, optional, default=None sample_weight: Weight of each sample of the training set.
        :param int or list, optional, default=None treat: If there is only one discrete treatment, then treat indicates the
            treatment group. If there are multiple treatment groups, then treat
            should be a list of str with length equal to the number of treatments. 
            For example, when there are multiple discrete treatments,
                
                array(['run', 'read'])
            
            means the treat value of the first treatment is taken as 'run' and
            that of the second treatment is taken as 'read'.
        :param int or list, optional, default=None control: See treat.
        
        :returns: Fitted CTCausalForest
        :rtype: instance of CTCausalForest

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