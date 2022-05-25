# TODO: add copyright of the implementation of sklearn tree
import numbers
from math import ceil
from copy import deepcopy

import numpy as np
# from ctypes import memset, sizeof, c_double, memmove

from sklearn.utils import check_random_state
from sklearn.utils import check_scalar

from sklearn.tree._splitter import Splitter
from sklearn.tree._splitter import BestSplitter
from sklearn.tree._splitter import RandomSplitter
from sklearn.tree._criterion import Criterion
# from sklearn.tree._criterion import MSE
from sklearn.tree._tree import Tree
from sklearn.tree._tree import DepthFirstTreeBuilder
from sklearn.tree._tree import BestFirstTreeBuilder

from ..utils import logging
from ..utils._common import convert2array

from ._tree.tree_criterion import PRegCriteria
from ._tree.tree_criterion import PRegCriteria1
from ..estimator_model._tree.tree_criterion import MSE

logger = logging.get_logger(__name__)


CRITERIA = {
    "policy_reg": PRegCriteria,
    "policy_clf": None,
    'policy_test': MSE,
    'policy_test1': PRegCriteria1
}

SPLITTERS = {
    "best": BestSplitter,
    "random": RandomSplitter,
}


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
        covariate, *,
        effect=None,
        effect_array=None,
        est_model=None,
        sample_weight=None,
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
        ce, v = convert2array(data, effect, covariate)
        self.covariate = covariate
        
        if effect_array is not None:
            ce = effect_array
        else:
            if effect is None:
                assert est_model is not None, \
                    'The causal effect is not provided and no estimator model is'
                'provided to estimate it from the data.'
                assert est_model.covariate == covariate

                if hasattr(est_model, 'covariate_transformer'):
                    self.cov_transformer = est_model.covariate_transformer
                    v = self.cov_transformer.transform(v)

                ce = est_model.estimate(data)

        self._v = v
        
        # check random state
        random_state = check_random_state(self.random_state)

        # Determine output settings
        n_samples, self.n_features_in = v.shape  # dimension of the input

        # reshape ce if necessary
        if ce.ndim == 1:
            ce = ce.reshape(-1, 1)

        self.n_outputs_ = ce.shape[1]

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
        if self.max_leaf_nodes is not None:
            check_scalar(
                self.max_leaf_nodes,
                name='max_leaf_nodes',
                target_type=numbers.Integral,
                min_val=2,
            )

        max_leaf_nodes = -1 if self.max_leaf_nodes is None else self.max_leaf_nodes

        # check min_weight_fraction_leaf
        check_scalar(
            self.min_weight_fraction_leaf,
            name='min_weight_fraction_leaf',
            target_type=numbers.Real,
            min_val=0.0,
            max_val=0.5,
        )

        # check max_features
        if isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                max_features = max(1, int(np.sqrt(self.n_features_in)))
            elif self.max_features == 'log2':
                max_features = max(1, int(np.log2(self.n_features_in)))
            else:
                raise ValueError(
                    'Invalid value for max_features. Allowed string values'
                    f'Allowed string values are "sqrt" or "log2", but was given {self.max_features}.'
                )
        elif self.max_features is None:
            max_features = self.n_features_in
        elif isinstance(self.max_features, numbers.Integral):
            check_scalar(
                self.max_features,
                name='max_features',
                target_type=numbers.Integral,
                min_val=1,
                include_boundaries='left',
            )
            max_features = self.max_features
        else:
            check_scalar(
                self.max_features,
                name='max_features',
                target_type=numbers.Real,
                min_val=0.0,
                max_val=1.0,
                include_boundaries='right',
            )
            if self.max_features > 0.0:
                max_features = max(
                    1, int(self.max_features * self.n_features_in)
                )
            else:
                max_features = 0

        self.max_features_ = max_features

        check_scalar(
            self.min_impurity_decrease,
            name='min_impurity_decrease',
            target_type=numbers.Real,
            min_val=0.0,
        )

        if len(ce) != n_samples:
            raise ValueError(
                f'The number of labels {len(ce)} does not match the number of samples'
            )

        # set min_weight_leaf
        if sample_weight is None:
            min_weight_leaf = self.min_weight_fraction_leaf * n_samples
        else:
            min_weight_leaf = self.min_weight_fraction_leaf * \
                np.sum(sample_weight)

        # Build tree step 1. Set up criterion
        criterion = self.criterion

        if not isinstance(criterion, Criterion):
            # assert criterion == 'policy_clf' or criterion == 'policy_reg', f'The str criterion should be policy, but was given {self.criterion}'
            criterion = CRITERIA[self.criterion](self.n_outputs_, n_samples)
        else:
            criterion = deepcopy(self.criterion)

        logger.info(
            f'Start building the causal tree with criterion {type(criterion).__name__}'
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

        logger.info(
            f'Building the causal tree with splitter {type(splitter).__name__}'
        )

        # Build tree step 3. Define the tree
        self.tree_ = Tree(
            self.n_features_in,
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

        logger.info(
            f'Building the causal tree with builder {type(builder).__name__}'
        )

        builder.build(self.tree_, v, ce, sample_weight)

        self._is_fitted = True

        return self

    def predict(self, data=None):
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
        pred = self._prepare4est(data)

        return pred

    def _prepare4est(self, data=None):
        assert self._is_fitted, 'The model is not fitted yet.'

        if data is None:
            v = self._v
        else:
            v = convert2array(data, self.covariate)[0]
            if hasattr(self, 'cov_transformer'):
                v = self.cov_transformer.transform(v)
        
        v = v.astype(np.float32)

        proba = self.tree_.predict(v)
        n_samples = v.shape[0]

        # if self.criterion == 'policy_reg':
        #     if self.n_outputs_ == 1:
        #         return proba[:, 0]
        #     else:
        #         return proba[:, :, 0]
        # else:
        #     pass

        return proba[:, :, 0]
        
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
