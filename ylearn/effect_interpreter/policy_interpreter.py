import numpy as np

from ._export import _PolicyTreeMPLExporter


class PolicyInterpreter:
    """
    Attributes
    ----------
    criterion : {'policy_reg'}, default to 'policy_reg' # TODO: may add more criterion

    n_outputs_ : int
        The number of outputs when fit() is performed.

    max_features_ : int
        The inferred value of max_features.

    n_features_in_ : int
        Number of features seen during fit().

    max_depth : int, default to None

    min_samples_split : int or float, default to 2

    min_samples_leaf : int or float, default to 1

    random_state : int

    max_leaf_nodes : int, default to None

    min_impurity_decrease : float, default to 0.0

    ccp_alpha : non-negative float, default to 0.0

    Methods
    ----------
    fit(data, outcome, treatment,
        adjustment=None, covariate=None, treat=None, control=None,)
        Fit the model on data.

    estimate(data=None, quantity=None)
        Estimate the causal effect of the treatment on the outcome in data.

    apply(v)
        Return the index of the leaf that each sample is predicted as.

    decision_path(v)
        Return the decision path.

    _prepare4est(data)
        Prepare for the estimation of the causal effect.

    Reference
    ----------
    This implementation is based on the implementation of BaseDecisionTree
    of sklearn.
    """

    def __init__(
        self, *,
        criterion='policy_reg',
        splitter='best',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=2022,
        max_leaf_nodes=None,
        max_features=None,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,
        min_weight_fraction_leaf=0.0,
    ):
        """
        Many parameters are similar to those of BaseDecisionTree of sklearn.

        Parameters
        ----------
        criterion : {'policy_reg'}, default to 'policy_reg' # TODO: may add more criterion
            The function to measure the quality of a split. The criterion for
            training the tree is (in the Einstein notation)
            
            .. math::
            
                    S = \sum_i g_{ik} y^k_{i},
                    
            where :math:`g_{ik} = \phi(v_i)_k` is a map from the covariates, :math:`v_i`, to a
            basis vector which has only one nonzero element in the :math:`R^k` space. By
            using this criterion, the aim of the model is to find the index of the
            treatment which will render the max causal effect, i.e., finding the
            optimal policy.

        splitter : {"best", "random"}, default="best"
            The strategy used to choose the split at each node. Supported
            strategies are "best" to choose the best split and "random" to choose
            the best random split.

        max_depth : int, default=None
            The maximum depth of the tree. If None, then nodes are expanded until
            all leaves are pure or until all leaves contain less than
            min_samples_split samples.

        min_samples_split : int or float, default=2
            The minimum number of samples required to split an internal node:
            - If int, then consider `min_samples_split` as the minimum number.
            - If float, then `min_samples_split` is a fraction and
            `ceil(min_samples_split * n_samples)` are the minimum
            number of samples for each split.

        min_samples_leaf : int or float, default=1
            The minimum number of samples required to be at a leaf node.
            A split point at any depth will only be considered if it leaves at
            least ``min_samples_leaf`` training samples in each of the left and
            right branches.  This may have the effect of smoothing the model,
            especially in regression.
            - If int, then consider `min_samples_leaf` as the minimum number.
            - If float, then `min_samples_leaf` is a fraction and
            `ceil(min_samples_leaf * n_samples)` are the minimum
            number of samples for each node.

        min_weight_fraction_leaf : float, default=0.0
            The minimum weighted fraction of the sum total of weights (of all
            the input samples) required to be at a leaf node. Samples have
            equal weight when sample_weight is not provided.

        max_features : int, float or {"sqrt", "log2"}, default=None
            The number of features to consider when looking for the best split:
            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
            `int(max_features * n_features)` features are considered at each
            split.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

        random_state : int
            Controls the randomness of the estimator.

        max_leaf_nodes : int, default to None
            Grow a tree with ``max_leaf_nodes`` in best-first fashion.
            Best nodes are defined as relative reduction in impurity.
            If None then unlimited number of leaf nodes.

        min_impurity_decrease : float, default=0.0
            A node will be split if this split induces a decrease of the impurity
            greater than or equal to this value.
            The weighted impurity decrease equation is the following::
                N_t / N * (impurity - N_t_R / N_t * right_impurity
                                    - N_t_L / N_t * left_impurity)
            where ``N`` is the total number of samples, ``N_t`` is the number of
            samples at the current node, ``N_t_L`` is the number of samples in the
            left child, and ``N_t_R`` is the number of samples in the right child.
            ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
            if ``sample_weight`` is passed.

        ccp_alpha : non-negative float, default to 0.0
            Value for pruning the tree. #TODO: not implemented yet.
        """
        self._is_fitted = False

        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features

        self.node_dict_ = None
        self.treatment_names_ = None


    def fit(
        self,
        data,
        est_model,
        *,
        covariate=None,
        treatment_names=None,
        effect=None,
        effect_array=None,
    ):
        """Fit the PolicyInterpreter model to interpret the policy for the causal
        effect estimated by the est_model on data.

        Parameters
        ----------
        data : pandas.DataFrame
            The input samples for the est_model to estimate the causal effects
            and for the CEInterpreter to fit.

        est_model : estimator_model
            est_model should be any valid estimator model of ylearn which was 
            already fitted and can estimate the CATE.
        """
        from ylearn.policy.policy_model import PolicyTree

        covariate = est_model.covariate if covariate is None else covariate
        # outcome = est_model.outcome

        assert covariate is not None, 'Need covariate to interpret the causal effect.'

        if est_model is not None:
            assert est_model._is_fitted
        
        self.covariate = covariate

        # y = convert2array(data, outcome)[0]
        # n, _y_d = y.shape
        # assert _y_d == 1

        self._tree_model = PolicyTree(
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            max_leaf_nodes=self.max_leaf_nodes,
            max_features=self.max_features,
            min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self.ccp_alpha,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
        )

        self._tree_model.fit(
            data=data,
            covariate=covariate,
            effect=effect,
            effect_array=effect_array,
            est_model=est_model
        )

        self._is_fitted = True
        self.treatment_names_ = treatment_names

        return self

    def decide(self, data):
        assert self._is_fitted, 'The model is not fitted yet. Please use the fit method first.'
        y_pred_ind = self._tree_model.predict_ind(data)  # y_pred_ind.shape=(n,)
        if self.treatment_names_ is not None:
            return np.array(self.treatment_names_).take(y_pred_ind)
        else:
            return y_pred_ind

    def predict(self, data):
        assert self._is_fitted, 'The model is not fitted yet. Please use the fit method first.'
        return self._tree_model.predict_opt_effect(data)

    def interpret(self, data=None):
        assert self._is_fitted, 'The model is not fitted yet. Please use the fit method first.'

        v = self._tree_model._check_features(v=None, data=data)

        policy_pred = self._tree_model.predict_opt_effect(data=data)
        policy_value = policy_pred.max(1)
        policy_ind = policy_pred.argmax(1)
        node_indicator = self._tree_model.decision_path(v=v)
        leaf_id = self._tree_model.apply(v=v)
        feature = self._tree_model.tree_.feature
        threshold = self._tree_model.tree_.threshold

        result_dict = {}
        
        for sample_id, sample in enumerate(v):
            result_dict[f'sample_{sample_id}'] = ''
            node_index = node_indicator.indices[
                node_indicator.indptr[sample_id]: node_indicator.indptr[sample_id + 1]
            ]

            for node_id in node_index:
                if leaf_id[sample_id] == node_id:
                    continue

                if v[sample_id, feature[node_id]] <= threshold[node_id]:
                    thre_sign = '<='
                else:
                    thre_sign = '>'

                feature_id = feature[node_id]
                value = v[sample_id, feature_id]
                thre = threshold[node_id]
                result_dict[f'sample_{sample_id}'] += f'decision node {node_id}: (covariate [{sample_id}, {feature_id}] = {value})' \
                    f' {thre_sign} {thre} \n'

            result_dict[f'sample_{sample_id}'] += f'The recommended policy is treatment {policy_ind[sample_id]} with value {policy_value[sample_id]}'
            
        return result_dict

    def plot(
        self, *,
        feature_names=None,
        max_depth=None,
        class_names=None,
        label='all',
        filled=True,
        node_ids=False,
        proportion=False,
        rounded=False,
        precision=3,
        ax=None,
        fontsize=None
    ):
        """Plot the tree model.
        The sample counts that are shown are weighted with any sample_weights that
        might be present.
        The visualization is fit automatically to the size of the axis.
        Use the ``figsize`` or ``dpi`` arguments of ``plt.figure``  to control
        the size of the rendering.

        Parameters
        ----------        
        max_depth : int, default=None
            The maximum depth of the representation. If None, the tree is fully
            generated.
        
        class_names : list of str or bool, default=None
            Names of each of the target classes in ascending numerical order.
            Only relevant for classification and not supported for multi-output.
            If ``True``, shows a symbolic representation of the class name.
        
        label : {'all', 'root', 'none'}, default='all'
            Whether to show informative labels for impurity, etc.
            Options include 'all' to show at every node, 'root' to show only at
            the top root node, or 'none' to not show at any node.
        
        filled : bool, default=False
            When set to ``True``, paint nodes to indicate majority class for
            classification, extremity of values for regression, or purity of node
            for multi-output.
        
        impurity : bool, default=True
            When set to ``True``, show the impurity at each node.
        
        node_ids : bool, default=False
            When set to ``True``, show the ID number on each node.
        
        proportion : bool, default=False
            When set to ``True``, change the display of 'values' and/or 'samples'
            to be proportions and percentages respectively.
        
        rounded : bool, default=False
            When set to ``True``, draw node boxes with rounded corners and use
            Helvetica fonts instead of Times-Roman.
        
        precision : int, default=3
            Number of digits of precision for floating point in the values of
            impurity, threshold and value attributes of each node.
        
        ax : matplotlib axis, default=None
            Axes to plot to. If None, use current axis. Any previous content
            is cleared.
        
        fontsize : int, default=None
            Size of text font. If None, determined automatically to fit figure.
        
        Returns
        -------
        annotations : list of artists
            List containing the artists for the annotation boxes making up the
            tree.
        """
        assert self._is_fitted

        if feature_names == None:
            feature_names = self.covariate

        exporter = _PolicyTreeMPLExporter(feature_names=feature_names,
                                          treatment_names=self.treatment_names_,
                                          max_depth=max_depth,
                                          filled=filled,
                                          impurity=False,
                                          rounded=rounded,
                                          precision=precision,
                                          fontsize=fontsize)

        exporter.export(self._tree_model, node_dict=self.node_dict_, ax=ax)
