import numbers
from math import ceil
from copy import deepcopy

import numpy as np
# from ctypes import memset, sizeof, c_double, memmove

from sklearn.utils import check_random_state
from sklearn.utils import check_scalar
from sklearn.tree._splitter import BestSplitter
from sklearn.tree._tree import (DepthFirstTreeBuilder, Tree,
                                BestFirstTreeBuilder)

from ylearn.utils import logging
from ylearn.utils._common import convert2array

logger = logging.get_logger(__name__)


class PolicyModel:
    """
    A class for finding the optimal policy for maxmizing the causal effect
    with the tree model.

    The criterion for training the tree is (in the Einstein notation)
        S = \sum_i g_{ik} y^k_{i},
    where g_{ik} = \phi(v_i)_k is a map from the covariates, v_i, to a basis vector
    which has only one nonzero element in the R^k space.

    Attributes
    ----------
    feature_importances_ : ndarray of shape (n_features, )
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
    categories : str, optional, default to 'auto'

    Methods
    ----------
    fit(data, outcome, treatment,
        adjustment=None, covariate=None, treat=None, control=None,)
        Fit the model on data.
    estimate(data=None, quantity=None)
        Estimate the causal effect of the treatment on the outcome in data.
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
    # TODO: sample_weight

    def __init__(
        self, *,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=2022,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,
        sample_weight=None,
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
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state
        self.sample_weight = sample_weight

    def fit(
        self,
        data,
        covariate,
        effect=None,
        effect_array=None,
        est_model=None,
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
        effect_array : ndarray
            ndarray of ausal effects.

        Returns
        ----------
        instance of self
        """
        if effect_array is not None:
            ce = effect_array
        else:
            ce, v = convert2array(data, effect, covariate)
            if effect is None:
                assert est_model is not None, \
                    'The causal effect is not provided and no estimator model is'
                'provided to estimate it from the data.'
                assert est_model.covariate == covariate

                if hasattr(est_model, 'covariate_transformer'):
                    v = est_model.covariate_transformer.transform(v)

                ce = est_model.estimate(data)

        # check random state
        random_state = check_random_state(self.random_state)

        # Determine output settings
        n_samples, n_features_in = v.shape  # dimension of the input

        # reshape ce if necessary
        if ce.ndim == 1:
            ce = ce.reshape(-1, 1)

        self.n_outputs = ce.shape[1]

        # Check parameters
        if self.max_depth is not None:
            check_scalar(
                self.max_depth,
                name='max_depth',
                target_type=numbers.Integral,
                min_val=1,
            )
        max_depth = np.iinfo(np.int32).max if self.max_depth is None \
            else self.max_depth

        # check self.min_samples_leaf
        if isinstance(self.min_samples_leaf, numbers.Integral):
            check_scalar(
                self.min_samples_leaf,
                name='min_samples_leaf',
                target_type=numbers.Integral,
                min_val=1,
            )
            min_samples_leaf = self.min_samples_leaf
        else:
            check_scalar(
                self.min_samples_leaf,
                name='min_samples_leaf',
                target_type=numbers.Real,
                min_val=0.0,
                include_boundaries='neither',
            )
            min_samples_leaf = int(ceil(self.min_samples_leaf * n_samples))

        # check self.min_samples_split
        if isinstance(self.min_samples_split, numbers.Integral):
            check_scalar(
                self.min_samples_split,
                name='min_samples_split',
                target_type=numbers.Integral,
                min_val=2,
            )
            min_samples_split = self.min_samples_split
        else:
            check_scalar(
                self.min_samples_split,
                name='min_samples_split',
                target_type=numbers.Real,
                min_val=0.0,
                max_val=1.0,
                include_boundaries='right',
            )
            min_samples_split = int(ceil(self.min_samples_split * n_samples))
            min_samples_split = max(2, min_samples_split)
        
        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        # check self.max_leaf_nodes
        max_leaf_nodes = -1 if self.max_leaf_nodes is None \
            else self.max_leaf_nodes

        if len(ce) != n_samples:
            raise ValueError(
                f'Number of labels {len(ce)} does not match number of samples'
            )

        min_weight_leaf = 0  # maybe this need more modifications

        # Build tree step 1. Set up criterion
        criterion = None

        logger.info(
            f'Start building the causal tree with criterion {type(criterion).__name__}'
        )

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

        builder.build(self.tree, v, ce, self.sample_weight)
        self._is_fitted = True

        return self

    def estimate(self, data=None):
        """Estimate the optimal policy for the causal effects of the treatment
        on the outcome in the data.

        Parameters
        ----------
        data : pandas.DataFrame, optional. Defaults to None
            If None, data will be set as the training data.    

        Returns
        -------
        ndarray or float, optional
            The estimated causal effect with the type of the quantity.
        """
        effect = self._prepare4est(data)
        pass

    def _prepare4est(self, data=None):
        raise NotImplemented()

    def _prune_tree(self):
        pass

    def apply(self, X):
        """Return the index of the leaf that each sample is predicted as.
        """
        pass

    def decision_path(self, X):
        pass

    @property
    def feature_importance(self):
        pass

    def plot_result(self,):
        pass
