*********************************
Policy: Selecting the Best Option
*********************************

In tasks such as policy evaluation, e.g., [Athey2020]_, besides the causal effects, we may also have interets in other questions such as whether an example should be assigned to a treament and if the answer is yes, which option is
the best among all possible treament values. YLearn implements :class:`PolicyTree` for such purpose. Given a trained estimator model or estimated causal effects, it finds the optimal polices for each
example by building a decision tree model which aims to maximize the causal effect of each example.

The criterion for training the tree is 

.. math::

    S = \sum_i\sum_k g_{ik}e_{ki}

where :math:`g_{ik} = \phi(v_i)_k` with :math:`\phi: \mathbb{R}^D \to \mathbb{R}^K` being a map from :math:`v_i\in \mathbb{R}^D` to a basis vector with only one nonzero element in :math:`\mathbb{R}^K` and :math:`e_{ki}` denotes
the causal effect of taking the :math:`k`-th value of the treatment for example :math:`i`.

.. seealso::

    :py:class:`BaseDecisionTree` in sklearn.

Note that one can use the :class:`PolicyInterpreter` to interpret the result of a policy model.

.. topic:: Example

    .. code-block:: python

        import numpy as np

        from ylearn.policy.policy_model import PolicyTree
        from ylearn.utils._common import to_df

        # build dataset
        v = np.random.normal(size=(1000, 10))
        y = np.hstack([v[:, [0]] < 0, v[:, [0]] > 0])

        data = to_df(v=v)
        covariate = data.columns

        est = PolicyTree(criterion='policy_reg')
        est.fit(data=data, covariate=covariate, effect_array=y.astype(float))
    
    >>> 06-23 14:53:14 I ylearn.p.policy_model.py 452 - Start building the policy tree with criterion PRegCriteria
    >>> 06-23 14:53:14 I ylearn.p.policy_model.py 468 - Building the policy tree with splitter BestSplitter
    >>> 06-23 14:53:14 I ylearn.p.policy_model.py 511 - Building the policy tree with builder DepthFirstTreeBuilder
    >>> <ylearn.policy.policy_model.PolicyTree at 0x7ff1ee5f2eb0>


Class Structures
================

.. py:class:: ylearn.policy.policy_model.PolicyTree(*, criterion='policy_reg', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=2022, max_leaf_nodes=None, max_features=None, min_impurity_decrease=0.0, ccp_alpha=0.0, min_weight_fraction_leaf=0.0)
    
    :param {'policy_reg'}, default="'policy_reg'" criterion: The function to measure the quality of a split. The criterion for
            training the tree is (in the Einstein notation)
            
            .. math::

                    S = \sum_i g_{ik} e^k_{i},
        
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

    .. py:method:: fit(data, covariate, *, effect=None, effect_array=None, est_modle=None, sample_weight=None)
        
        Fit the PolicyInterpreter model to interpret the policy for the causal
        effect estimated by the est_model on data. One has several options for passing the causal effects, which usually is a vector of (n, j, i)
        where `n` is the number of the examples, `j` is the dimension of the outcome, and `i` is the number of possible treatment values or the dimension of the treatment:
            
            1. Only pass `est_model`. Then `est_model` will be used to generate the causal effects.

            2. Only pass `effect_array` which will be set as the causal effects and `effect` and `est_model` will be ignored.

            3. Only pass `effect`. This usually is a list of names of the causal effect in `data` which will then be used as the causal effects for training the model.

        :param pandas.DataFrame data: The input samples for the est_model to estimate the causal effects
            and for the CEInterpreter to fit.
        :param estimator_model est_model: est_model should be any valid estimator model of ylearn which was 
            already fitted and can estimate the CATE. If `effect=None` and `effect_array=None`, then `est_model` can not be None and the causal
            effect will be estimated by the `est_model`.
        :param list of str, optional, default=None covariate: Names of the covariate. 
        :param list of str, optional, default=None effect: Names of the causal effect in `data`. If `effect_array` is not None, then `effect` will be ignored.
        :param numpy.ndarray, default=None effect_array: The causal effect that waited to be fitted by  :class:`PolicyTree`. If this is not provided and `est_model` is None, then `effect` can not be None.

        :returns: Fitted PolicyModel
        :rtype: instance of PolicyModel

    .. py:method:: predict_ind(data=None)

        Estimate the optimal policy for the causal effects of the treatment
        on the outcome in the data, i.e., return the index of the optimal treatment.

        :param pandas.DataFrame, optional, default=None data: The test data in the form of the DataFrame. The model will only use this if v is set as None. In this case, if data is also None, then the data used for trainig will be used.

        :returns: The index of the optimal treatment dimension.
        :rtype: ndarray or int, optional

    .. py:method:: predict_opt_effect(data=None)

        Estimate the value of the optimal policy for the causal effects of the treatment
        on the outcome in the data, i.e., return the value of the causal effects
        when taking the optimal treatment.

        :param pandas.DataFrame, optional, default=None data: The test data in the form of the DataFrame. The model will only use this if v is set as None. In this case, if data is also None, then the data used for trainig will be used.

        :returns: The estimated causal effect with the optimal treatment value.
        :rtype: ndarray or float, optional

    .. py:method:: apply(*, v=None, data=None)

        Return the index of the leaf that each sample is predicted as.
        
        :param numpy.ndarray, default=None v: The input samples as an ndarray. If None, then the DataFrame data
            will be used as the input samples.
        :param pandas.DataFrame, default=None data: The input samples. The data must contains columns of the covariates
            used for training the model. If None, the training data will be
            passed as input samples.

        :returns: For each datapoint v_i in v, return the index of the leaf v_i
            ends up in. Leaves are numbered within ``[0; self.tree_.node_count)``, possibly with gaps in the
            numbering.
        :rtype: v_leaves : array-like of shape (n_samples, )

    .. py:method:: decision_path(*, v=None, data=None)

        Return the decision path.

        :param numpy.ndarray, default=None v: The input samples as an ndarray. If None, then the DataFrame data
            will be used as the input samples.
        :param pandas.DataFrame, default=None data: The input samples. The data must contains columns of the covariates
            used for training the model. If None, the training data will be
            passed as input samples.

        :returns: Return a node indicator CSR matrix where non zero elements
            indicates that the samples goes through the nodes.
        :rtype: indicator : sparse matrix of shape (n_samples, n_nodes)

    .. py:method:: get_depth()

        Return the depth of the policy tree.
        The depth of a tree is the maximum distance between the root
        and any leaf.

        :returns: The maximum depth of the tree.
        :rtype: int
    
    .. py:method:: get_n_leaves()

        Return the number of leaves of the policy tree.

        :returns: Number of leaves
        :rtype: int
    
    .. py:property:: feature_importance

        Return the feature importances.
        The importance of a feature is computed as the (normalized) total
        reduction of the criterion brought by that feature.
        It is also known as the Gini importance.
        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

        :returns: Normalized total reduction of criteria by feature
            (Gini importance).
        :rtype: ndarray of shape (n_features,)

    .. py:property:: n_features_

        :returns: number of features
        :rtype: int

    .. py:method:: plot(*, feature_names=None, max_depth=None, class_names=None, label='all', filled=False, node_ids=False, proportion=False, rounded=False, precision=3, ax=None, fontsize=None)

        Plot the PolicyTree.
        The sample counts that are shown are weighted with any sample_weights that
        might be present.
        The visualization is fit automatically to the size of the axis.
        Use the ``figsize`` or ``dpi`` arguments of ``plt.figure``  to control
        the size of the rendering.

        :returns: List containing the artists for the annotation boxes making up the
            tree.
        :rtype: annotations : list of artists
