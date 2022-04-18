import numpy as np

from sklearn.utils.validation import check_is_fitted
from sklearn.tree._splitter import BestSplitter
from sklearn.tree._tree import DepthFirstTreeBuilder, Tree

from .tree_criterion import CMSE


class CausalTree:
    # TODO: add confidence interval
    """
    A class for estimating causal effect with decision tree.

    Attributes
    ----------
    feature_importances_ : ndarray of shape (n_features,)
        The feature importances.
    tree_ : Tree instance
        The underlying Tree object.
    max_depth : int, default to None
    min_samples_split : int or float, default to 2
    min_samples_leaf : int or float, default to 1
    random_state : int
    max_leaf_nodes : int, default to None
    min_impurity_decrease : float, default to 0.0
    ccp_alpha : non-negative float, default to 0.0

    Methods
    ----------
    fit()
    predict()
    estimate()
    apply()
    """

    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,
    ):
        """
        Many parameters are similar to those of BaseDecisionTree of sklearn.

        Parameters
        ----------
        max_depth : int, default to None
            The maximum depth of the tree.
        min_samples_split : int or float, default to 2
            The minimum number of samples required to split an internal node.
        min_samples_leaf : int or float, default to 1
            The minimum number of samples required to be at a leaf node.
        random_state : int
        max_leaf_nodes : int, default to None
            Grow a tree with ``max_leaf_nodes`` in best-first fashion.
            Best nodes are defined as relative reduction in impurity.
            If None then unlimited number of leaf nodes.
        min_impurity_decrease : float, default to 0.0
            A node will be split if this split induces a decrease of the
            impurity greater than or equal to this value.
        ccp_alpha : non-negative float, default to 0.0
            Value for pruning the tree. #TODO: not implemented yet.

        See Also
        --------
        BaseDecisionTree : The default implementation of decision tree.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.tree_ = None

    def fit(self, data, outcome, treatment, treatment_value=1):
        """Fit the model to data.

        Parameters
        ----------
        data : pandas.DataFrame
            Currently only support this. #TODO: try to support ndarray.
        outcome : str or list of str
            Name of the outcome.
        treatment : str or list of str
            Name of the treatment vector.
        treatment_value : int, optional, defaults to 1
            If treatment == treatment_value for a data point, then it is in
            the treatment group, otherwise it is in the control group.

        Returns:
            self: The fitted causal tree.
        """
        # TODO: currently only support binary treatment, try to find if we can
        # support discrete treatment.
        # Determine output settings
        y = data[outcome]  # outcome vector
        t = data[treatment]  # treatment vector
        X = data.drop([outcome, treatment], axis=1)  # covariates
        y = np.atleast_1d(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_samples, n_features_in = X.shape  # dimension of the input
        n_outputs = 1

        # Check parameters and etermine output settings
        sample_weight = (t == treatment_value).astype(int)
        min_weight_leaf = 2  # maybe this needs more modifications

        # Build tree step 1. Set up criterion
        criterion = CMSE(1, n_samples)

        # Build tree step 2. Define splitter
        splitter = BestSplitter(
            criterion,
            n_features_in,
            self.min_samples_leaf,
            min_weight_leaf,
            self.random_state,
        )

        # Build tree step 3. Define the tree
        self.tree_ = Tree(
            n_features_in,
            np.array([1] * n_outputs, dtype=np.intp),
            n_outputs,
        )

        # Build tree step 3. Build the tree
        # TODO: try to be more compatible with the sklearn BasBaseDecisionTree
        builder = DepthFirstTreeBuilder(
            splitter,
            self.min_samples_split,
            self.min_samples_leafmin_samples_leaf,
            min_weight_leaf,
            self.max_depth,
            self.min_impurity_decrease,
        )
        builder.build(self.tree_, X, y, sample_weight)

        return self

    def predict(self, data, outcome=None, treatment=None):
        if treatment is not None:
            data = data.drop(treatment, axis=1)
        if outcome is not None:
            data = data.drop(outcome, axis=1)

        check_is_fitted(self)
        proba = self.tree_.predict(data)
        return proba[:, 0]

    def estimate(self, data, outcome, treatment, treatment_value=1):
        """Based on my current understanding, the causal tree only solves
        estimation of heterogeneous causal effects. Thus we may not need
        estimations for other effects.
        """
        self.fit(data, outcome, treatment, treatment_value)
        result = self.predict(data, outcome, treatment)
        return result

    def _prune_tree(self):
        pass

    def apply(self, X):
        """Return the index of the leaf that each sample is predicted as.
        """
        check_is_fitted(self)
        return self.tree_.apply(X)

    @property
    def feature_importance(self):
        pass
