# some parts of this code are forked from sci-kit learn

import numbers
from numbers import Integral, Real
from math import ceil

import numpy as np

from sklearn.utils import check_random_state
from sklearn.utils.validation import _check_sample_weight

from ylearn.sklearn_ex.cloned.tree._tree import Tree

from ._splitter import GrfTreeBestSplitter
from ._tree import GrfTreeBestFirstBuilder
from ...utils import convert2array, get_wv
from ...base_models import BaseEstModel
from ._criterion import GrfTreeCriterion

from ylearn.sklearn_ex.cloned.tree import _tree

DOUBLE = _tree.DOUBLE


class GrfTree(BaseEstModel):
    """A class for estimating causal effect with decision tree.

    Parameters
    ----------
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

    max_features : int, float or {"sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split:

            1. If int, then consider `max_features` features at each split.
            2. If float, then `max_features` is a fraction and `int(max_features * n_features)` features are considered at each split.
            3. If "sqrt", then `max_features=sqrt(n_features)`.
            4. If "log2", then `max_features=log2(n_features)`.
            5. If None, then `max_features=n_features`.

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
        Value for pruning the tree. *Not implemented yet*.

    categories : str, optional. Defaults to 'auto'.

    See Also
    --------
    BaseDecisionTree : The default implementation of decision tree in sklearn.

    Methods
    ----------
    fit(data, outcome, treatment, adjustment=None, covariate=None, treat=None, control=None)
        Fit the model on data.

        Parameters
        ----------
        data : pandas.DataFrame

        outcome : str or list of str
            Names of the outcomes.

        treatment : str or list of str
            Names of the treatment vectors.

        covariate : str of list of str
            Names of the covariate vectors.

        adjustment : str of list of str
            Names of the covariate vectors. Note that we may only need the covariate
            set, which usually is a subset of the adjustment set.

        treat : int or list, optional
            If there is only one discrete treament, then treat indicates the
            treatment group. If there are multiple treatment groups, then treat
            should be a list of str with length equal to the number of treatments.
            For example, when there are multiple discrete treatments,
                array(['run', 'read'])
            means the treat value of the first treatment is taken as 'run' and
            that of the second treatment is taken as 'read'.

        control : int or list, optional
            See treat for more information

        Returns
        ----------
        instance of CausalTree
            The fitted CausalTree.

    estimate(data=None, quantity=None)
        Estimate the causal effect of the treatment on the outcome in data.

        Parameters
        ----------
        data : pandas.DataFrame, optional, by default None
            If None, data will be set as the training data.

        quantity : str, optional, by default None
            The type of the causal effect. Avaliable options are:

                1. 'CATE' : the estimator will evaluate the CATE;
                2. 'ATE' : the estimator will evaluate the ATE;
                3. None : the estimator will evaluate the CITE.

        Returns
        -------
        ndarray or float, optional
            The estimated causal effect with the type of the quantity.

    get_depth()
        Return the depth of the causal tree. The depth of a tree is the maximum distance between the root
        and any leaf.

        Returns
        -------
        self.tree_.max_depth : int
            The maximum depth of the tree.

    get_n_leaves()
        Return the number of leaves of the causal tree.

        Returns
        -------
        self.tree_.n_leaves : int
            Number of leaves.

    apply(*, data=None, wv=None)
        Return the index of the leaf that each sample is predicted as.

        Parameters
        ----------
        wv : ndarray
            The input samples as an ndarray. If None, then the DataFrame data
            will be used as the input samples.

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

    decision_path(*, data=None, wv=None)
        Return the decision path.

        Parameters
        ----------
        wv : ndarray
            The input samples as an ndarray. If None, then the DataFrame data
            will be used as the input samples.

        data : DataFrame, optional
            The input samples. The data must contains columns of the covariates
            used for training the model. If None, the training data will be
            passed as input samples , by default None

        Returns
        -------
        indicator : sparse matrix of shape (n_samples, n_nodes)
            Return a node indicator CSR matrix where non zero elements
            indicates that the samples goes through the nodes.

    feature_importance()
        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            Normalized total reduction of criteria by feature
            (Gini importance).

    plot_causal_tree(max_depth=None, feature_names=None, label='all', filled=False, node_ids=False, proportion=False, rounded=False, precision=3, ax=None, fontsize=None)
        Plot a causal tree.
        The visualization is fit automatically to the size of the axis.
        Use the ``figsize`` or ``dpi`` arguments of ``plt.figure``  to control
        the size of the rendering.

        Parameters
        ----------
        max_depth : int, default=None
            The maximum depth of the representation. If None, the tree is fully
            generated.

        label : {'all', 'root', 'none'}, default='all'
            Whether to show informative labels for impurity, etc.
            Options include 'all' to show at every node, 'root' to show only at
            the top root node, or 'none' to not show at any node.

        filled : bool, default=False
            When set to ``True``, paint nodes to indicate majority class for
            classification, extremity of values for regression, or purity of node
            for multi-output.

        feature_names : list, default=None
            Names of features. If None, then the names of the adjustment or covariate
            will be used.

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

    def __init__(
        self,
        *,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=2022,
        max_leaf_nodes=None,
        max_features=None,
        min_impurity_decrease=0.0,
        min_weight_fraction_leaf=0.0,
        ccp_alpha=0.0,
        categories="auto",
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha

    def _check_data(self, data, outcome, treatment, adjustment, covariate):
        """Return transformed data in the form of array.

        Parameters
        ----------
        data : pd.DataFrame
            _description_
        outcome : str or list of str
            Names of outcome
        treatment : str or list of str
            Names of treatment
        adjustment : str or list of str
            Names of adjustment set, by default None
        covariate : str or list of str
            Names of covariat set, by default None
        """
        pass

    def fit(
        self,
        data,
        outcome,
        treatment,
        adjustment=None,
        covariate=None,
    ):
        # Note that this check_data should return arrays with at least 2 dimensions
        x, y, w, v = self.check_data(data, outcome, treatment, adjustment, covariate)

        if y.ndim > 1:
            assert y.shape[1] == 1, f"Currently only support scalar outcome"
        y = y.squeeze()

        # TODO: may consider add intercept to treatment matrix x

        self._fit_with_array(x, y, w, v)

        return self

    def predict(self, data=None):
        assert self._is_fitted, "The model is not fitted yet"
        w, v = self.check_data(data, self.covariate)
        return self._predict_with_array(w, v)

    def _fit_with_array(self, x, y, w, v, sample_weight=None):
        # TODO: clarify the role of w and v, currently we treat all w as v
        """
        Parameters
        ----------
        x : :py:class:`ndarray` of shape `(n, p)`
            The treatment vector of the training data of `n` examples, each with `p` dimensions, e.g., p-dimension one-hot vector if discrete treatment is assumed
        y : :py:class:`ndarray` of shape `(n,)`
            The outcome vector of the training data of `n` examples
        w : :py:class:`ndarray` of shape `(n, p)`
            The adjustment vector aka confounders of the training data
        v : :py:class:`ndarray` of shape `(n, p)`
            The covariate vector of the training data specifying hetergeneity
        """
        random_state = check_random_state(self.random_state)
        wv = get_wv(w, v)
        n_samples, self.n_features_in_ = wv.shape

        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.n_outputs_ = y.shape[1]
        self.d_treatments = x.shape[1]

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)
        if getattr(x, "dtype", None) != DOUBLE or not x.flags.contiguous:
            x = np.ascontiguousarray(x, dtype=DOUBLE)

        max_depth = np.iinfo(np.int32).max if self.max_depth is None else self.max_depth

        if isinstance(self.min_samples_leaf, numbers.Integral):
            min_samples_leaf = self.min_samples_leaf
        else:  # float
            min_samples_leaf = int(ceil(self.min_samples_leaf * n_samples))

        if isinstance(self.min_samples_split, numbers.Integral):
            min_samples_split = self.min_samples_split
        else:  # float
            min_samples_split = int(ceil(self.min_samples_split * n_samples))
            min_samples_split = max(2, min_samples_split)

        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                max_features = self.n_features_in_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_in_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_in_)))
        elif self.max_features is None:
            max_features = self.n_features_in_
        elif isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1, int(self.max_features * self.n_features_in_))
            else:
                max_features = 0

        self.max_features_ = max_features

        max_leaf_nodes = 3 if self.max_leaf_nodes is None else self.max_leaf_nodes

        if len(y) != n_samples or len(x) != n_samples:
            raise ValueError(
                f"Number of labels={len(y)} or treatments={len(x)} does not match number of samples={n_samples}"
            )

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, wv, DOUBLE)

        # Set min_weight_leaf from min_weight_fraction_leaf
        if sample_weight is None:
            min_weight_leaf = self.min_weight_fraction_leaf * n_samples
        else:
            min_weight_leaf = self.min_weight_fraction_leaf * np.sum(sample_weight)

        # Build tree
        criterion = GrfTreeCriterion(self.n_outputs_, n_samples, self.d_treatments)
        splitter = GrfTreeBestSplitter(
            criterion,
            self.max_features_,
            min_samples_leaf,
            min_weight_leaf,
            random_state,
        )
        self.tree_ = Tree(
            self.n_features_in_,
            np.array([1] * self.n_outputs_, dtype=np.intp),
            self.n_outputs_,
        )
        builder = GrfTreeBestFirstBuilder(
            splitter,
            min_samples_split,
            min_samples_leaf,
            min_weight_leaf,
            max_depth,
            max_leaf_nodes,
            self.min_impurity_decrease,
        )
        builder.build_ex(self.tree_, wv, y, x, sample_weight)

        return self

    def _predict_with_array(self, w, v, return_node=False):
        pass
