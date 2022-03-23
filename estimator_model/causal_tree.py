import numbers
from estimator_model.utils import sample

import numpy as np

from abc import ABCMeta
from abc import abstractclassmethod
from math import ceil

from sklearn.tree._splitter import BestSplitter
from sklearn.tree._tree import DepthFirstTreeBuilder, Tree

from .tree_criterion import CMSE


class CausalTree:
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
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.tree_ = None

    def fit(self, data, outcome, treatment, treatment_value=1):
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
        sample_weight = (t == treatment).astype(int)
        min_weight_leaf = 2  # maybe this needs more modifications
        max_depth = 1

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
            max_depth,
            self.min_impurity_decrease,
        )
        builder.build(self.tree_, X, y, sample_weight)

        return self

    def predict(self):
        pass

    def estimate(self):
        pass

    def _prune_tree(self):
        pass

    def apply(self):
        pass
