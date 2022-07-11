# TODO: add copyright of the implementation of sklearn tree
import numbers
import pandas

from math import ceil
from copy import deepcopy

import numpy as np

# import pydotplus
# from ctypes import memset, sizeof, c_double, memmove

import sklearn

from sklearn.utils import check_random_state
from sklearn.utils import check_scalar

from ylearn.sklearn_ex.cloned.tree._splitter import Splitter
from ylearn.sklearn_ex.cloned.tree._splitter import BestSplitter
from ylearn.sklearn_ex.cloned.tree._splitter import RandomSplitter
from ylearn.sklearn_ex.cloned.tree._criterion import Criterion

# from ylearn.sklearn_ex.cloned.tree._criterion import MSE
from ylearn.sklearn_ex.cloned.tree._tree import Tree
from ylearn.sklearn_ex.cloned.tree._tree import DepthFirstTreeBuilder
from ylearn.sklearn_ex.cloned.tree._tree import BestFirstTreeBuilder
from sklearn.tree import plot_tree

from ..utils import logging
from ..utils._common import convert2array

from ._tree.tree_criterion import PRegCriteria
from ._tree.tree_criterion import PRegCriteria1
from ..estimator_model._tree.tree_criterion import MSE

logger = logging.get_logger(__name__)


CRITERIA = {
    "policy_reg": PRegCriteria,
    "policy_test": MSE,
    "policy_test1": PRegCriteria1,
}

SPLITTERS = {
    "best": BestSplitter,
    "random": RandomSplitter,
}


class PolicyTree:
    """
    A class for finding the optimal policy for maxmizing the causal effect
    with the tree model.

    The criterion for training the tree is (in the Einstein notation)
        S = \sum_i g_{ik} y^k_{i},
    where g_{ik} = \phi(v_i)_k is a map from the covariates, v_i, to a basis vector
    which has only one nonzero element in the R^k space.

    Attributes
    ----------
    criterion : {'policy_reg'}, default to 'policy_reg' # TODO: may add more criterion

    feature_importances_ : ndarray of shape (n_features,)
        The feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the
        (normalized) total reduction of the criterion brought by that feature.

    tree_ : Tree instance
        The underlying Tree object. Please refer to
        ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object and
        :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
        for basic usage of these attributes.

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
        Estimate the value of the optimal policy for the causal effects of the treatment
        on the outcome in the data, i.e., return the value of the causal effects
        when taking the optimal treatment.

    predict_ind(data=None,)
        Estimate the optimal policy for the causal effects of the treatment
        on the outcome in the data, i.e., return the index of the optimal treatment.

    apply(X)
        Return the index of the leaf that each sample is predicted as.

    decision_path(X)
        Return the decision path.

    _prepare4est(data)
        Prepare for the estimation of the causal effect.

    Reference
    ----------
    This implementation is based on the implementation of BaseDecisionTree
    of sklearn.
    """

    def __init__(
        self,
        *,
        criterion="policy_reg",
        splitter="best",
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
                    S = \sum_i g_{ik} y^k_{i},
            where g_{ik} = \phi(v_i)_k is a map from the covariates, v_i, to a
            basis vector which has only one nonzero element in the R^k space. By
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

    def fit(
        self,
        data,
        covariate,
        *,
        effect=None,
        effect_array=None,
        est_model=None,
        sample_weight=None,
        **kwargs,
    ):
        # TODO: consider possibility for generalizing continuous treatment
        """Fit the model on data.

        Parameters
        ----------
        data : pandas.DataFrame

        covariate : str or list of str
            Names of the covariates.

        effect : str or list of str
            Names of the causal effects vectors.

        effect_array : ndarray of shape (n, n_treatments)
            The causal effect that waited to be fitted by  :class:`PolicyTree`. If this is not provided and `est_model` is None, then `effect` can not be None.

        est_model : Any valid estimator model in ylearn
            If `effect=None` and `effect_array=None`, then `est_model` can not be None and the causal
            effect will be estimated by the `est_model`.

        Returns
        ----------
        instance of self
        """
        ce, v = convert2array(data, effect, covariate)
        self.covariate = covariate

        if effect_array is not None:
            ce = effect_array
        else:
            if effect is None:
                assert (
                    est_model is not None
                ), "The causal effect is not provided and no estimator model is"
                "provided to estimate it from the data."

                assert est_model._is_fitted

                # assert est_model.covariate == covariate
                self.covariate = est_model.covariate
                v = convert2array(data, self.covariate)[0]

                if (
                    hasattr(est_model, "covariate_transformer")
                    and est_model.covariate_transformer is not None
                ):
                    self.cov_transformer = est_model.covariate_transformer
                    v = self.cov_transformer.transform(v)

                ce = est_model.effect_nji(data)
                ce = self._check_ce(ce)

        self._v = v

        # check random state
        random_state = check_random_state(self.random_state)

        # Determine output settings
        n_samples, self.n_features_in_ = v.shape  # dimension of the input

        # reshape ce if necessary
        if ce.ndim == 1:
            ce = ce.reshape(-1, 1)

        self.n_outputs_ = ce.shape[1]

        # Check parameters
        if self.max_depth is not None:
            check_scalar(
                self.max_depth,
                name="max_depth",
                target_type=numbers.Integral,
                min_val=1,
            )
        max_depth = np.iinfo(np.int32).max if self.max_depth is None else self.max_depth

        # check self.min_samples_leaf
        if isinstance(self.min_samples_leaf, numbers.Integral):
            check_scalar(
                self.min_samples_leaf,
                name="min_samples_leaf",
                target_type=numbers.Integral,
                min_val=1,
            )
            min_samples_leaf = self.min_samples_leaf
        else:
            check_scalar(
                self.min_samples_leaf,
                name="min_samples_leaf",
                target_type=numbers.Real,
                min_val=0.0,
                include_boundaries="neither",
            )
            min_samples_leaf = int(ceil(self.min_samples_leaf * n_samples))

        # check self.min_samples_split
        if isinstance(self.min_samples_split, numbers.Integral):
            check_scalar(
                self.min_samples_split,
                name="min_samples_split",
                target_type=numbers.Integral,
                min_val=2,
            )
            min_samples_split = self.min_samples_split
        else:
            check_scalar(
                self.min_samples_split,
                name="min_samples_split",
                target_type=numbers.Real,
                min_val=0.0,
                max_val=1.0,
                include_boundaries="right",
            )
            min_samples_split = int(ceil(self.min_samples_split * n_samples))
            min_samples_split = max(2, min_samples_split)

        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        # check self.max_leaf_nodes
        if self.max_leaf_nodes is not None:
            check_scalar(
                self.max_leaf_nodes,
                name="max_leaf_nodes",
                target_type=numbers.Integral,
                min_val=2,
            )

        max_leaf_nodes = -1 if self.max_leaf_nodes is None else self.max_leaf_nodes

        # check min_weight_fraction_leaf
        check_scalar(
            self.min_weight_fraction_leaf,
            name="min_weight_fraction_leaf",
            target_type=numbers.Real,
            min_val=0.0,
            max_val=0.5,
        )

        # check max_features
        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_in_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_in_)))
            else:
                raise ValueError(
                    "Invalid value for max_features. Allowed string values"
                    f'Allowed string values are "sqrt" or "log2", but was given {self.max_features}.'
                )
        elif self.max_features is None:
            max_features = self.n_features_in_
        elif isinstance(self.max_features, numbers.Integral):
            check_scalar(
                self.max_features,
                name="max_features",
                target_type=numbers.Integral,
                min_val=1,
                include_boundaries="left",
            )
            max_features = self.max_features
        else:
            check_scalar(
                self.max_features,
                name="max_features",
                target_type=numbers.Real,
                min_val=0.0,
                max_val=1.0,
                include_boundaries="right",
            )
            if self.max_features > 0.0:
                max_features = max(1, int(self.max_features * self.n_features_in_))
            else:
                max_features = 0

        self.max_features_ = max_features

        check_scalar(
            self.min_impurity_decrease,
            name="min_impurity_decrease",
            target_type=numbers.Real,
            min_val=0.0,
        )

        if len(ce) != n_samples:
            raise ValueError(
                f"The number of labels {len(ce)} does not match the number of samples"
            )

        # set min_weight_leaf
        if sample_weight is None:
            min_weight_leaf = self.min_weight_fraction_leaf * n_samples
        else:
            min_weight_leaf = self.min_weight_fraction_leaf * np.sum(sample_weight)

        # Build tree step 1. Set up criterion
        criterion = self.criterion

        if not isinstance(criterion, Criterion):
            # assert criterion == 'policy_clf' or criterion == 'policy_reg', f'The str criterion should be policy, but was given {self.criterion}'
            criterion = CRITERIA[self.criterion](self.n_outputs_, n_samples)
        else:
            criterion = deepcopy(self.criterion)

        logger.info(
            f"Start building the policy tree with criterion {type(criterion).__name__}"
        )

        # Build tree step 2. Define splitter
        splitter = self.splitter

        if not isinstance(self.splitter, Splitter):
            splitter = SPLITTERS[self.splitter](
                criterion,
                self.max_features_,
                min_samples_leaf,
                min_weight_leaf,
                random_state,
            )

        logger.info(f"Building the policy tree with splitter {type(splitter).__name__}")

        # Build tree step 3. Define the tree
        self.tree_ = Tree(
            self.n_features_in_,
            np.array([1] * self.n_outputs_, dtype=np.intp),
            self.n_outputs_,
        )

        # Build tree step 3. Build the tree
        if max_leaf_nodes < 0:
            builder = DepthFirstTreeBuilder(
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                self.min_impurity_decrease,
            )
        else:
            builder = BestFirstTreeBuilder(
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                max_leaf_nodes,
                self.min_impurity_decrease,
            )

        logger.info(f"Building the policy tree with builder {type(builder).__name__}")

        builder.build(self.tree_, v, ce, sample_weight)

        self._is_fitted = True

        return self

    def predict_ind(self, data=None):
        """Estimate the optimal policy for the causal effects of the treatment
        on the outcome in the data, i.e., return the index of the optimal treatment.

        Parameters
        ----------
        data : pandas.DataFrame, optional. Defaults to None
            If None, data will be set as the training data.

        Returns
        -------
        ndarray or float, optional
            The index of the optimal treatment dimension.
        """
        pred = self._prepare4est(data).argmax(1)

        return pred

    def predict_opt_effect(self, data=None):
        """Estimate the value of the optimal policy for the causal effects of the treatment
        on the outcome in the data, i.e., return the value of the causal effects
        when taking the optimal treatment.

        Parameters
        ----------
        data : pandas.DataFrame, optional. Defaults to None
            If None, data will be set as the training data.

        Returns
        -------
        ndarray or float, optional
            The estimated causal effect with the optimal treatment value.
        """
        pred = self._prepare4est(data)

        return pred

    def _prepare4est(self, data=None):
        assert self._is_fitted, "The model is not fitted yet."

        v = self._check_features(v=None, data=data)

        proba = self.tree_.predict(v)
        n_samples = v.shape[0]

        if self.criterion == "policy_reg":
            # TODO: consider more about continuous treatment
            if self.n_outputs_ == 1:
                return proba[:, 0]
            else:
                return proba[:, :, 0]
        else:
            raise ValueError(f"The type of criterion {self.criterion} is not supported")

    def _prune_tree(self):
        raise NotImplemented()

    def _check_ce(self, ce):
        if ce.ndim == 1:
            ce = ce.reshape(-1, 1)
        else:
            n, _yx_d = ce.shape[0], ce.shape[1]
            if ce.ndim == 3:
                assert (
                    _yx_d == 1
                ), f"Expect effect array with shape {(n, ce.shape[2])}, but was given {(n, _yx_d, ce.shape[2])}"

                ce = ce.reshape(n, -1)
            elif ce.ndim == 2:
                pass
            else:
                raise ValueError(
                    f"Expect effect array with shape (num_samples, num_treatments) but was given {ce.shape}"
                )

        return ce

    def _check_features(self, *, v=None, data=None):
        """Validate the training data on predict (probabilities)."""

        if v is not None:
            v = v.reshape(-1, 1) if v.ndim == 1 else v
            assert v.shape[1] == self.n_features_in_
            v = v.astype(np.float32)

            return v

        if data is None:
            v = self._v
        else:
            assert isinstance(data, pandas.DataFrame)

            v = convert2array(data, self.covariate)[0]
            if hasattr(self, "cov_transformer"):
                v = self.cov_transformer.transform(v)

            assert v.shape[1] == self.tree_.n_features

        v = v.astype(np.float32)

        return v

    def apply(self, *, v=None, data=None):
        """Return the index of the leaf that each sample is predicted as.

        Parameters
        ----------
        data : DataFrame, optional
            The input samples. The data must contains columns of the covariates
            used for training the model. If None, the training data will be
            passed as input samples , by default None

        Returns
        -------
        v_leaves : array-like of shape (n_samples, )
            For each datapoint v_i in v, return the index of the leaf v_i
            ends up in. Leaves are numbered within
            ``[0; self.tree_.node_count)``, possibly with gaps in the
            numbering.
        """
        assert self._is_fitted, "The model is not fitted yet."

        v = self._check_features(v=v, data=data)

        return self.tree_.apply(v)

    def decision_path(self, *, v=None, data=None):
        """Return the decision path in the tree.

        Parameters
        ----------
        data : DataFrame, optional
            The input samples. The data must contains columns of the covariates
            used for training the model. If None, the training data will be
            passed as input samples , by default None

        Returns
        -------
        indicator : sparse matrix of shape (n_samples, n_nodes)
            Return a node indicator CSR matrix where non zero elements
            indicates that the samples goes through the nodes.
        """
        assert self._is_fitted, "The model is not fitted yet."

        v = self._check_features(v=v, data=data)

        return self.tree_.decision_path(v)

    @property
    def feature_importance(self):
        """Return the feature importances.
        The importance of a feature is computed as the (normalized) total
        reduction of the criterion brought by that feature.
        It is also known as the Gini importance.
        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            Normalized total reduction of criteria by feature
            (Gini importance).
        """
        assert self._is_fitted, "The model is not fitted yet."

        return self.tree_.compute_feature_importances()

    def get_depth(self):
        """Return the depth of the policy tree.
        The depth of a tree is the maximum distance between the root
        and any leaf.
        Returns
        -------
        self.tree_.max_depth : int
            The maximum depth of the tree.
        """
        assert self._is_fitted
        return self.tree_.max_depth

    def get_n_leaves(self):
        """Return the number of leaves of the policy tree.
        Returns
        -------
        self.tree_.n_leaves : int
            Number of leaves.
        """
        assert self._is_fitted
        return self.tree_.n_leaves

    def plot(
        self,
        *,
        max_depth=None,
        feature_names=None,
        class_names=None,
        label="all",
        filled=False,
        node_ids=False,
        proportion=False,
        rounded=False,
        precision=3,
        ax=None,
        fontsize=None,
    ):
        """Plot a policy tree.
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

        impurity = False
        feature_names = self.covariate if feature_names is None else feature_names

        return plot_tree(
            self,
            max_depth=max_depth,
            feature_names=feature_names,
            class_names=class_names,
            label=label,
            filled=filled,
            impurity=impurity,
            node_ids=node_ids,
            proportion=proportion,
            rounded=rounded,
            precision=precision,
            ax=ax,
            fontsize=fontsize,
        )

    @property
    def n_features_(self):
        return self.n_features_in_
