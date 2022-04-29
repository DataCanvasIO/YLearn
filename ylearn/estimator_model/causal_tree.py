import numpy as np
from copy import deepcopy

# from ctypes import memset, sizeof, c_double, memmove

from sklearn.utils import check_random_state
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree._splitter import BestSplitter
from sklearn.tree._tree import (DepthFirstTreeBuilder, Tree,
                                BestFirstTreeBuilder)

from .utils import (convert2array, get_wv, get_treat_control)
from .base_models import BaseEstLearner
from .tree_criterion import CMSE, MSE, HonestCMSE
# import pyximport
# pyximport.install(setup_args={"script_args": ["--verbose"]})


class CausalTree(BaseEstLearner):
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

    Reference
    ----------
    https://arxiv.org/abs/1504.01132
    """
    # TODO: sample_weight

    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=2022,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,
        eps=1e-5,
        categories='auto'
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
        # self.categories = categories
        # self.random_state = random_state
        self.eps = eps

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha

        super().__init__(
            random_state=random_state,
            categories=categories,
        )

    def fit(
        self,
        data,
        outcome,
        treatment,
        adjustment=None,
        covariate=None,
        treat=None,
        control=None,
    ):
        # TODO: consider possibility for generalizing continuous treatment
        """Fit the model to data.

        Parameters
        ----------
        data : pandas.DataFrame
        outcome : str or list of str
            Names of the outcomes.
        treatment : str or list of str
            Names of the treatment vectors..

        Returns
        ----------
            self: The fitted causal tree.
        """
        assert adjustment is not None or covariate is not None, \
            'Need adjustment or covariate to perform estimation.'
        super().fit(data, outcome, treatment,
                    adjustment=adjustment,
                    covariate=covariate,
                    )
        random_state = check_random_state(self.random_state)
        y, x, w, v = convert2array(
            data, outcome, treatment, adjustment, covariate
        )
        wv = get_wv(w, v)

        # Determin treatment settings
        if self.categories == 'auto' or self.categories is None:
            categories = 'auto'
        else:
            categories = list(self.categories)

        self.transformer = OrdinalEncoder(categories=categories)
        self.transformer.fit(x)
        x = self.transformer.transform(x)
        n_treatments = len(self.transformer.categories_)

        # get new dataset with treat and controls
        treat = get_treat_control(treat, n_treatments, True)
        control = get_treat_control(control, n_treatments, treat=False)
        # TODO: this should be much more simpler when considering single treat
        _tr = np.all(treat == x, axis=1)
        _crtl = np.all(control == x, axis=1)
        label = np.any((_tr, _crtl), axis=0)
        y = y[label]
        wv = wv[label]
        sample_weight = _tr[label].astype(int)

        # Determine output settings
        n_samples, n_features_in = wv.shape  # dimension of the input
        self.n_outputs = y.shape[1]
        self._wv = wv

        # Check parameters and etermine output settings
        # TODO: check self.max_depth
        max_depth = np.iinfo(np.int32).max if self.max_depth is None \
            else self.max_depth

        # check self.min_samples_leaf
        min_samples_leaf = self.min_samples_leaf

        # check self.min_samples_split
        min_samples_split = self.min_samples_split
        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        # check self.max_leaf_nodes
        max_leaf_nodes = -1 if self.max_leaf_nodes is None \
            else self.max_leaf_nodes

        if len(y) != n_samples:
            raise ValueError(
                f'Number of labels {len(y)} does not match number of samples'
            )

        min_weight_leaf = 0  # maybe this need more modifications

        # Build tree step 1. Set up criterion
        # criterion = deepcopy(MSE(self.n_outputs, n_samples))
        # criterion = deepcopy(CMSE(self.n_outputs, n_samples))
        criterion = deepcopy(HonestCMSE(self.n_outputs, n_samples))

        # Build tree step 2. Define splitter
        splitter = BestSplitter(
            criterion,
            n_features_in,
            min_samples_leaf,
            min_weight_leaf,
            random_state,
        )

        # Build tree step 3. Define the tree
        self.tree = Tree(
            n_features_in,
            np.array([1] * self.n_outputs, dtype=np.intp),
            self.n_outputs,
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

        eps = self.eps
        builder.build(self.tree, wv, y, sample_weight + eps)
        self._is_fitted = True

        return self

    def estimate(self, data=None, quantity=None):
        effect = self._prepare4est(data)

        if quantity == 'ATE' or quantity == 'CATE':
            np.mean(effect, axis=0)
        else:
            return effect

    def _prepare4est(self, data=None):
        if not self._is_fitted:
            raise Exception('The model is not fitted yet.')

        if data is None:
            wv = self._wv
        else:
            w, v = convert2array(data, self.adjustment, self.covariate)
            wv = get_wv(w, v)
        wv = wv.astype(np.float32)

        effect = self.tree.predict(wv)
        return effect

    def _prune_tree(self):
        pass

    def apply(self, X):
        """Return the index of the leaf that each sample is predicted as.
        """
        return self.tree.apply(X)

    def decision_path(self, X):
        return self.tree.decision_path(X)

    @property
    def feature_importance(self):
        return self.tree.compute_feature_importances()


# class _CausalTreeOld:
#     """
#     A class for estimating causal effect with decision tree.

#     Attributes
#     ----------
#     feature_importances_ : ndarray of shape (n_features,)
#         The feature importances.
#     tree_ : Tree instance
#         The underlying Tree object.
#     max_depth : int, default to None
#     min_samples_split : int or float, default to 2
#     min_samples_leaf : int or float, default to 1
#     random_state : int
#     max_leaf_nodes : int, default to None
#     min_impurity_decrease : float, default to 0.0
#     ccp_alpha : non-negative float, default to 0.0

#     Methods
#     ----------
#     fit()
#     predict()
#     estimate()
#     apply()
#     """

#     def __init__(
#         self,
#         max_depth=None,
#         min_samples_split=2,
#         min_samples_leaf=1,
#         random_state=None,
#         max_leaf_nodes=None,
#         min_impurity_decrease=0.0,
#         ccp_alpha=0.0,
#     ):
#         """
#         Many parameters are similar to those of BaseDecisionTree of sklearn.

#         Parameters
#         ----------
#         max_depth : int, default to None
#             The maximum depth of the tree.
#         min_samples_split : int or float, default to 2
#             The minimum number of samples required to split an internal node.
#         min_samples_leaf : int or float, default to 1
#             The minimum number of samples required to be at a leaf node.
#         random_state : int
#         max_leaf_nodes : int, default to None
#             Grow a tree with ``max_leaf_nodes`` in best-first fashion.
#             Best nodes are defined as relative reduction in impurity.
#             If None then unlimited number of leaf nodes.
#         min_impurity_decrease : float, default to 0.0
#             A node will be split if this split induces a decrease of the
#             impurity greater than or equal to this value.
#         ccp_alpha : non-negative float, default to 0.0
#             Value for pruning the tree. #TODO: not implemented yet.

#         See Also
#         --------
#         BaseDecisionTree : The default implementation of decision tree.
#         """
#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.min_samples_leaf = min_samples_leaf
#         self.random_state = random_state
#         self.max_leaf_nodes = max_leaf_nodes
#         self.min_impurity_decrease = min_impurity_decrease
#         self.ccp_alpha = ccp_alpha
#         self.tree = None

#     def fit(
#         self,
#         data,
#         outcome,
#         treatment,
#         adjustment=None,
#         covariate=None,
#     ):
#         """Fit the model to data.

#         Parameters
#         ----------
#         data : pandas.DataFrame
#             Currently only support this.
#         outcome : str or list of str
#             Name of the outcome.
#         treatment : str or list of str
#             Name of the treatment vector.
#         treatment_value : int, optional, defaults to 1
#             If treatment == treatment_value for a data point, then it is in
#             the treatment group, otherwise it is in the control group.

#         Returns:
#             self: The fitted causal tree.
#         """
#         assert adjustment is not None or covariate is not None, \
#             'Need adjustment or covariate to perform estimation.'

#         # Determine output settings
#         self.outcome = outcome
#         self.treatment = treatment
#         self.adjustment = adjustment
#         self.covariate = covariate

#         n = len(data)
#         y, x, w, v = convert2array(
#             data, outcome, treatment, adjustment, covariate
#         )

#         if y.ndim == 1:
#             y = y.reshape(-1, 1)
#         n_samples, n_features_in = X.shape  # dimension of the input
#         n_outputs = 1

#         # Check parameters and etermine output settings
#         sample_weight = (t == treatment_value).astype(int)
#         min_weight_leaf = 2  # maybe this needs more modifications

#         # Build tree step 1. Set up criterion
#         criterion = CMSE(1, n_samples)

#         # Build tree step 2. Define splitter
#         splitter = BestSplitter(
#             criterion,
#             n_features_in,
#             self.min_samples_leaf,
#             min_weight_leaf,
#             self.random_state,
#         )

#         # Build tree step 3. Define the tree
#         self.tree = Tree(
#             n_features_in,
#             np.array([1] * n_outputs, dtype=np.intp),
#             n_outputs,
#         )

#         # Build tree step 3. Build the tree
#         # TODO: try to be more compatible with the sklearn BasBaseDecisionTree
#         builder = DepthFirstTreeBuilder(
#             splitter,
#             self.min_samples_split,
#             self.min_samples_leafmin_samples_leaf,
#             min_weight_leaf,
#             self.max_depth,
#             self.min_impurity_decrease,
#         )
#         builder.build(self.tree, X, y, sample_weight)

#         return self

#     def predict(self, data, outcome=None, treatment=None):
#         if treatment is not None:
#             data = data.drop(treatment, axis=1)
#         if outcome is not None:
#             data = data.drop(outcome, axis=1)

#         check_is_fitted(self)
#         proba = self.tree.predict(data)
#         return proba[:, 0]

#     def estimate(self, data, outcome, treatment, treatment_value=1):
#         """Based on my current understanding, the causal tree only solves
#         estimation of heterogeneous causal effects. Thus we may not need
#         estimations for other effects.
#         """
#         self.fit(data, outcome, treatment, treatment_value)
#         result = self.predict(data, outcome, treatment)
#         return result

#     def _prune_tree(self):
#         pass

#     def apply(self, X):
#         """Return the index of the leaf that each sample is predicted as.
#         """
#         check_is_fitted(self)
#         return self.tree.apply(X)

#     @property
#     def feature_importance(self):
#         pass
