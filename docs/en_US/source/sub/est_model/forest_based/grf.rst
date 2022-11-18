******************************************************
Generalized Random Forest for Causal Effect Estimation 
******************************************************

To adpat random forest to causal effect estimation, [Athey2018]_ proposed a generalized version of it, named as Generalized Random Forest (GRF), by altering the criterion
when building a single tree and designing a new kind of ensemble method to combine these trained trees. GRF can be used in, for example, quantile regression while in YLearn,
we focus on its ability of performing highly flexible non-parametric causal effect estimation.

We now consider such estimation with GRF. Suppose that we observe samples :math:`(X_i, Y_i, V_i) \in \mathbb{R}^{d_x} \times \mathbb{R} \times \mathbb{R}^{d_v}` where :math:`Y`
is the outcome, :math:`X` is the treatment and :math:`V` is the covariate which ensures the unconfoundness condition. The forest weights :math:`\alpha_i(v)` is defined by

.. math::

    \alpha_i^b(v) = \frac{\mathbb{I}\left( \left\{ V_i \in L^b(v) \right\} \right)}{|L^b(v)|},\\
    \alpha_i(v) = \frac{1}{B} \sum_{b = 1}^B \alpha_i^b(v),

where the subscript :math:`b` refers to the :math:`b`-th tree with a total number of :math:`B` such trees, :math:`L^b(v)` is the leaf that the sample with covariate :math:`v`
belongs to, and :math:`|L^b(v)|` denotes the total number of training samples which fall into the samel leaf as the sample :math:`v` for the :math:`b`-th tree. Then the estimated
causal effect can be expressed by

.. math::

    \left( \sum_{i=1}^n \alpha_i(x)(X_i - \bar{X}_\alpha)(X_i - \bar{X}_\alpha)^T\right)^{-1} \sum_{i = 1}^n \alpha_i(v) (X_i - \bar{X}_\alpha)(Y_i - \bar{Y}_\alpha)

where :math:`\bar{X}_\alpha = \sum \alpha_i X_i` and :math:`\bar{Y}_\alpha = \sum \alpha_i Y_i`.

Besides the GRF, YLearn also implements a naive version of GRF with pure python in an easy to understand manner to help users get some insights on how GRF works in code level.
It is worth to mention that, however, this naive version of GRF is super slow (~5mins for fitting 100 trees in a dataset with 2000 samples and 10 features). One can find this naive
GRF in the folder ylearn/estimator_model/_naive_forest/. 

The formal version of GRF is summarized as follows.

Class Structures
================

.. py:class:: ylearn.estimator_model.GRForest(*, splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=2022, max_leaf_nodes=None, max_features=None, min_impurity_decrease=0.0, min_weight_fraction_leaf=0.0, ccp_alpha=0.0, categories='auto')

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

    :param str, optional, default='auto' categories: 

    .. py:method:: fit(data, outcome, treatment, adjustment=None, covariate=None, treat=None, control=None)
        
        Fit the model on data to estimate the causal effect.

        :param pandas.DataFrame data: The input samples for the est_model to estimate the causal effects
            and for the CEInterpreter to fit.
        :param list of str, optional outcome: Names of the outcomes.
        :param list of str, optional treatment: Names of the treatments.
        :param list of str, optional, default=None covariate: Names of the covariate vectors.
        :param list of str, optional, default=None adjustment: Names of the covariate vectors. Note that we may only need the covariate
            set, which usually is a subset of the adjustment set.
        :param int or list, optional, default=None treat: If there is only one discrete treatment, then treat indicates the
            treatment group. If there are multiple treatment groups, then treat
            should be a list of str with length equal to the number of treatments. 
            For example, when there are multiple discrete treatments,
                
                array(['run', 'read'])
            
            means the treat value of the first treatment is taken as 'run' and
            that of the second treatment is taken as 'read'.
        :param int or list, optional, default=None control: See treat.
        
        :returns: Fitted CausalTree
        :rtype: instance of CausalTree

    .. py:method:: estimate(data=None, quantity=None)

        Estimate the causal effect of the treatment on the outcome in data.

        :param pandas.DataFrame, optional, default=None data: If None, data will be set as the training data.
        :param str, optional, default=None quantity: Option for returned estimation result. The possible values of quantity include:
                
                1. *'CATE'* : the estimator will evaluate the CATE;
                
                2. *'ATE'* : the estimator will evaluate the ATE;
                
                3. *None* : the estimator will evaluate the ITE or CITE.

        :returns: The estimated causal effect with the type of the quantity.
        :rtype: ndarray or float, optional

    .. py:method:: plot_causal_tree(feature_names=None, max_depth=None, class_names=None, label='all', filled=False, node_ids=False, proportion=False, rounded=False, precision=3, ax=None, fontsize=None)

        Plot a policy tree.
        The sample counts that are shown are weighted with any sample_weights that
        might be present.
        The visualization is fit automatically to the size of the axis.
        Use the ``figsize`` or ``dpi`` arguments of ``plt.figure``  to control
        the size of the rendering.

        :returns: List containing the artists for the annotation boxes making up the
            tree.
        :rtype: annotations : list of artists
    
    .. py:method:: decision_path(*, data=None, wv=None)

        Return the decision path.

        :param numpy.ndarray, default=None wv: The input samples as an ndarray. If None, then the DataFrame data
            will be used as the input samples.
        :param pandas.DataFrame, default=None data: The input samples. The data must contains columns of the covariates
            used for training the model. If None, the training data will be
            passed as input samples.

        :returns: Return a node indicator CSR matrix where non zero elements
            indicates that the samples goes through the nodes.
        :rtype: indicator : sparse matrix of shape (n_samples, n_nodes)

    .. py:method:: apply(*, data=None, wv=None)

        Return the index of the leaf that each sample is predicted as.
        
        :param numpy.ndarray, default=None wv: The input samples as an ndarray. If None, then the DataFrame data
            will be used as the input samples.
        :param pandas.DataFrame, default=None data: The input samples. The data must contains columns of the covariates
            used for training the model. If None, the training data will be
            passed as input samples.

        :returns: For each datapoint v_i in v, return the index of the leaf v_i
            ends up in. Leaves are numbered within ``[0; self.tree_.node_count)``, possibly with gaps in the
            numbering.
        :rtype: v_leaves : array-like of shape (n_samples, )

    .. py:property:: feature_importance

        :returns: Normalized total reduction of criteria by feature (Gini importance).
        :rtype: ndarray of shape (n_features,)