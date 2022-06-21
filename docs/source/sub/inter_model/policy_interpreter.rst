.. _policy_int:

*****************
PolicyInterpreter
*****************

:class:`PolicyInterpreter` can be used to interpret the policy returned by an instance of :class:`PolicyTree`. By assigning
different strategies to different examples, it aims to maximize the casual effects of a subgroup and separate them from those 
with negative causal effects. 

Class Structures
================

.. py:class:: ylearn.interpreter.policy_interpreter.PolicyInterpreter(*, criterion='policy_reg', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=2022, max_leaf_nodes=None, max_features=None, min_impurity_decrease=0.0, ccp_alpha=0.0, min_weight_fraction_leaf=0.0)

    :param {'policy_reg'}, default="'policy_reg'" criterion: The function to measure the quality of a split. The criterion for
            training the tree is (in the Einstein notation)
            
            .. math::

                    S = \sum_i g_{ik} y^k_{i},
        
            where :math:`g_{ik} = \phi(v_i)_k` is a map from the covariates, :math:`v_i`, to a
            basis vector which has only one nonzero element in the :math:`R^k` space. By
            using this criterion, the aim of the model is to find the index of the
            treatment which will render the max causal effect, i.e., finding the
            optimal policy. 
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

    .. py:method:: fit(data, est_model, *, covariate=None, effect=None, effect_array=None)
        
        Fit the PolicyInterpreter model to interpret the policy for the causal
        effect estimated by the est_model on data.

        :param pandas.DataFrame data: The input samples for the est_model to estimate the causal effects
            and for the CEInterpreter to fit.
        :param estimator_model est_model: est_model should be any valid estimator model of ylearn which was 
            already fitted and can estimate the CATE.
        :param list of str, optional, default=None covariate: Names of the covariate. 
        :param list of str, optional, default=None effect: Names of the causal effect in `data`. If `effect_array` is not None, then `effect` will be ignored.
        :param numpy.ndarray, default=None effect_array: The causal effect that waited to be interpreted by the :class:`PolicyInterpreter`. If this is not provided, then `effect` can not be None.

        :returns: Fitted PolicyInterpreter
        :rtype: instance of PolicyInterpreter

    .. py:method:: interpret(*, data=None)

        Interpret the fitted model in the test data.

        :param pandas.DataFrame, optional, default=None data: The test data in the form of the DataFrame. The model will only use this if v is set as None. In this case, if data is also None, then the data used for trainig will be used.

        :returns: The interpreted results for all examples.
        :rtype: dict

    .. py:method:: plot(*, feature_names=None, max_depth=None, class_names=None, label='all', filled=False, node_ids=False, proportion=False, rounded=False, precision=3, ax=None, fontsize=None)

        Plot the tree model.
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