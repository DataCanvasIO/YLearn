.. _ce_int:

*************
CEInterpreter
*************

For the CATE :math:`\tau(v)` estimated by an estimator model, e.g., double machine learning model, :class:`CEInterpreter` interprets the results
by building a decision tree to model the relationships between :math:`\tau(v)` and the covariates :math:`v`. Then one can use the decision rules 
of the fitted tree model to analyze :math:`\tau(v)`.

Class Structures
================

.. py:class:: ylearn.effect_interpreter.ce_interpreter.CEInterpreter(*, criterion='squared_error', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=2022, max_leaf_nodes=None, max_features=None, min_impurity_decrease=0.0, min_weight_fraction_leaf=0.0, ccp_alpha=0.0, categories='auto')

    :param {"squared_error", "friedman_mse", "absolute_error",  "poisson"}, default="squared_error" criterion: The function to measure the quality of a split. Supported criteria
        are "squared_error" for the mean squared error, which is equal to
        variance reduction as feature selection criterion and minimizes the L2
        loss using the mean of each terminal node, "friedman_mse", which uses
        mean squared error with Friedman's improvement score for potential
        splits, "absolute_error" for the mean absolute error, which minimizes
        the L1 loss using the median of each terminal node, and "poisson" which
        uses reduction in Poisson deviance to find splits.        
    :param {"best", "random"}, default="best" splitter: The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.
    :param int, default=None max_depth: The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
    :param int or float, default=2 min_samples_split: The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and `ceil(min_samples_split * n_samples)` are the minimum number of samples for each split.
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
    :param int, default to None max_leaf_nodes: Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
    :param float, default=0.0 min_impurity_decrease: A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        The weighted impurity decrease equation is the following
            
            N_t / N * (impurity - N_t_R / N_t * right_impurity - N_t_L / N_t * left_impurity)
        
        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.
        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    .. py:method:: fit(data, est_model, **kwargs)
        
        Fit the CEInterpreter model to interpret the causal effect estimated
        by the est_model on data.

        :param pandas.DataFrame data: The input samples for the est_model to estimate the causal effects
            and for the CEInterpreter to fit.
        :param estimator_model est_model: est_model should be any valid estimator model of ylearn which was 
            already fitted and can estimate the CATE.
        
        :returns: Fitted CEInterpreter
        :rtype: instance of CEInterpreter

    .. py:method:: interpret(*, v=None, data=None)

        Interpret the fitted model in the test data.

        :param numpy.ndarray, optional, default=None v: The test covariates in the form of ndarray. If this is given, then data will be ignored and the model will use this as the test data.
        :param pandas.DataFrame, optional, default=None data: The test data in the form of the DataFrame. The model will only use this if v is set as None. In this case, if data is also None, then the data used for training will be used.

        :returns: The interpreted results for all examples.
        :rtype: dict

    .. py:method:: plot(*, feature_names=None, max_depth=None, class_names=None, label='all', filled=False, node_ids=False, proportion=False, rounded=False, precision=3, ax=None, fontsize=None)

        Plot the fitted tree model.
        The sample counts that are shown are weighted with any sample_weights that
        might be present.
        The visualization is fit automatically to the size of the axis.
        Use the ``figsize`` or ``dpi`` arguments of ``plt.figure``  to control
        the size of the rendering.

        :returns: List containing the artists for the annotation boxes making up the
            tree.
        :rtype: annotations : list of artists
    
.. topic:: Example

    pass