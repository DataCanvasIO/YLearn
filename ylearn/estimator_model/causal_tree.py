import numbers
from math import ceil
from copy import deepcopy

import numpy as np
import pandas

# from ctypes import memset, sizeof, c_double, memmove

import sklearn
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar

from sklearn.preprocessing import OrdinalEncoder

from ylearn.sklearn_ex.cloned.tree._splitter import Splitter
from ylearn.sklearn_ex.cloned.tree._splitter import BestSplitter
from ylearn.sklearn_ex.cloned.tree._splitter import RandomSplitter
from ylearn.sklearn_ex.cloned.tree._tree import DepthFirstTreeBuilder
from ylearn.sklearn_ex.cloned.tree._tree import BestFirstTreeBuilder
from ylearn.sklearn_ex.cloned.tree._tree import Tree
from sklearn.tree import plot_tree

from ..utils import logging
from .utils import (convert2array, get_wv, get_treat_control)
from .base_models import BaseEstModel

from ylearn.estimator_model._tree.tree_criterion import CMSE, MSE, HonestCMSE
# import pyximport
# pyximport.install(setup_args={"script_args": ["--verbose"]})

logger = logging.get_logger(__name__)

SPLITTERS = {"best": BestSplitter, "random": RandomSplitter}

EPS = 1e-5


class CausalTree(BaseEstModel):
    # TODO: add support for multi-output causal tree
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
    # TODO: sample_weight

    def __init__(
        self,
        *,
        splitter='best',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=2022,
        max_leaf_nodes=None,
        max_features=None,
        min_impurity_decrease=0.0,
        min_weight_fraction_leaf=0.0,
        ccp_alpha=0.0,
        categories='auto'
    ):
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
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
        """Fit the model on data.

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
            The fitted causal tree.
        """
        assert adjustment is not None or covariate is not None, \
            'Need adjustment or covariate to perform estimation.'
        
        # check random state
        random_state = check_random_state(self.random_state)

        super().fit(
            data, outcome, treatment,
            adjustment=adjustment,
            covariate=covariate,
        )
        
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
        treat = get_treat_control(treat, self.transformer, True)
        control = get_treat_control(control, self.transformer, treat=False)

        self.treat = treat
        self.control = control

        # TODO: this should be much more simpler when considering single treat
        _tr = np.all(treat == x, axis=1)
        _crtl = np.all(control == x, axis=1)
        label = np.any((_tr, _crtl), axis=0)
        y = y[label]
        wv = wv[label]
        sample_weight = _tr[label].astype(int)

        check_scalar(
            self.ccp_alpha,
            name='ccp_alpha',
            target_type=numbers.Real,
            min_val=0.0,
        )        
        
        # Determine output settings
        n_samples, self.n_features_in_ = wv.shape  # dimension of the input
        self._wv = wv

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.n_outputs_ = y.shape[1]

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
                max_features = max(1, int(np.sqrt(self.n_features_in_)))
            elif self.max_features == 'log2':
                max_features = max(1, int(np.log2(self.n_features_in_)))
            else:
                raise ValueError(
                    'Invalid value for max_features. Allowed string values'
                    f'Allowed string values are "sqrt" or "log2", but was given {self.max_features}.'
                )
        elif self.max_features is None:
            max_features = self.n_features_in_
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
                    1, int(self.max_features * self.n_features_in_)
                )
            else:
                max_features = 0

        self.max_features_ = max_features

        if len(y) != n_samples:
            raise ValueError(
                f'Number of labels {len(y)} does not match number of samples'
            )

        # set min_weight_leaf
        if sample_weight is None:
            min_weight_leaf = self.min_weight_fraction_leaf * n_samples
        else:
            min_weight_leaf = self.min_weight_fraction_leaf * \
                np.sum(sample_weight)

        # Build tree step 1. Set up criterion
        # criterion = deepcopy(MSE(self.n_outputs_, n_samples))
        # criterion = deepcopy(CMSE(self.n_outputs_, n_samples))
        criterion = deepcopy(HonestCMSE(self.n_outputs_, n_samples))

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

        logger.info(
            f'Building the causal tree with builder {type(builder).__name__}'
        )

        builder.build(self.tree_, wv, y, sample_weight + EPS)

        self._is_fitted = True

        return self

    def estimate(self, data=None, quantity=None):
        """Estimate the causal effect of the treatment on the outcome in the data.

        Parameters
        ----------
        data : pandas.DataFrame, optional, by default None
            If None, data will be set as the training data.
        
        quantity : str, optional, by default None
            The type of the causal effect. Avaliable options are:
                'CATE' : the estimator will evaluate the CATE;
                'ATE' : the estimator will evaluate the ATE;
                None : the estimator will evaluate the CITE.          

        Returns
        -------
        ndarray or float, optional
            The estimated causal effect with the type of the quantity.
        """
        if not self._is_fitted:
            raise Exception('The model is not fitted yet.')

        effect = self._prepare4est(data)

        logger.info(
            f'Start estimating the causal effect with the type of {quantity}.'
        )
        
        if quantity == 'ATE' or quantity == 'CATE':
            np.mean(effect, axis=0)
        else:
            return effect

    def _check_features(self, *, wv=None, data=None):
        """Validate the data for the model to estimate the causal effect.

        Parameters
        ----------
        wv : ndarray
            The test samples as an ndarray. If None, then the DataFrame data
            will be used as the test samples.
        
        data : pandas.DataFrame
            The test samples.

        Returns
        -------
        ndarray
            Valid input to the causal tree.
        """
        if wv is not None:
            wv = wv.reshape(-1, 1) if wv.ndim == 1 else wv
            assert wv.shape[1] == self.n_features_in_
            wv = wv.astype(np.float32)
            
            return wv

        if data is None:
            wv = self._wv
        else:
            assert isinstance(data, pandas.DataFrame)
            
            w, v = convert2array(data, self.adjustment, self.covariate)
            wv = get_wv(w, v)
        
        wv = wv.astype(np.float32)
        
        return wv

    def effect_nji(self, data=None):
        y_nji = self._prepare4est(data=data)
        n, y_d = y_nji.shape
        y_nji = y_nji.reshape(n, y_d, -1)
        zeros_ = np.zeros((n, y_d, 1))
        y_nji = np.concatenate((zeros_, y_nji), axis=2)
        return y_nji

    def _prepare4est(self, data=None):
        wv = self._check_features(wv=None, data=data)
        effect = self.tree_.predict(wv)

        return effect

    def _prune_tree(self):
        pass
    
    def get_depth(self):
        """Return the depth of the causal tree.
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
        """Return the number of leaves of the causal tree.
        Returns
        -------
        self.tree_.n_leaves : int
            Number of leaves.
        """
        assert self._is_fitted
        return self.tree_.n_leaves


    def apply(self, *, data=None, wv=None):
        """Return the index of the leaf that each sample is predicted as.

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
        """
        assert self._is_fitted, 'The model is not fitted yet.'
        wv = self._check_features(wv=wv, data=data)
        
        return self.tree_.apply(wv)     

    def decision_path(self, *, data=None, wv=None):
        """Return the decision path in the tree.

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
        """
        assert self._is_fitted, 'The model is not fitted yet.'

        v = self._check_features(wv=wv, data=data)

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
        assert self._is_fitted, 'The model is not fitted yet.'
        
        return self.tree_.compute_feature_importances()
    
    @property
    def n_features_(self):
        return self.n_features_in_
    
    def plot_causal_tree(
        self, *,
        max_depth=None,
        feature_names=None,
        label='all',
        filled=False,
        node_ids=False,
        proportion=False,
        rounded=False,
        precision=3,
        ax=None,
        fontsize=None
    ):
        """Plot a causal tree.
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
        assert self._is_fitted

        impurity = False

        if feature_names is None:
            feature_names = []
            
            if self.adjustment is not None:
                if isinstance(self.adjustment, str):
                    adj = [self.adjustment]
                else:
                    adj = list(self.adjustment)
                feature_names.extend(adj)
            
            if self.covariate is not None:
                if isinstance(self.covariate, str):
                    cov = [self.covariate]
                else:
                    cov = list(self.covariate)
                feature_names.extend(cov)

        return plot_tree(
            self,
            max_depth=max_depth,
            feature_names=feature_names,
            class_names=None,
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

# class _CausalTreeOld:
#     """
#     A class for estimating causal effect with decision tree.

#     Attributes
#     ----------
#     feature_importances_ : ndarray of shape (n_features,)
#         The feature importances.
#     tree_ : Tree instance
#         The underlying Tree object.
#     max_depth : int, by default None
#     min_samples_split : int or float, by default 2
#     min_samples_leaf : int or float, by default 1
#     random_state : int
#     max_leaf_nodes : int, by default None
#     min_impurity_decrease : float, by default 0.0
#     ccp_alpha : non-negative float, by default 0.0

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
#         max_depth : int, by default None
#             The maximum depth of the tree.
#         min_samples_split : int or float, by default 2
#             The minimum number of samples required to split an internal node.
#         min_samples_leaf : int or float, by default 1
#             The minimum number of samples required to be at a leaf node.
#         random_state : int
#         max_leaf_nodes : int, by default None
#             Grow a tree with ``max_leaf_nodes`` in best-first fashion.
#             Best nodes are defined as relative reduction in impurity.
#             If None then unlimited number of leaf nodes.
#         min_impurity_decrease : float, by default 0.0
#             A node will be split if this split induces a decrease of the
#             impurity greater than or equal to this value.
#         ccp_alpha : non-negative float, by default 0.0
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
#         self.tree_ = None

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
#         n_samples, self.n_features_in_ = X.shape  # dimension of the input
#         n_outputs = 1

#         # Check parameters and etermine output settings
#         sample_weight = (t == treatment_value).astype(int)
#         min_weight_leaf = 2  # maybe this needs more modifications

#         # Build tree step 1. Set up criterion
#         criterion = CMSE(1, n_samples)

#         # Build tree step 2. Define splitter
#         splitter = BestSplitter(
#             criterion,
#             self.n_features_in_,
#             self.min_samples_leaf,
#             min_weight_leaf,
#             self.random_state,
#         )

#         # Build tree step 3. Define the tree
#         self.tree_ = Tree(
#             self.n_features_in_,
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
#         builder.build(self.tree_, X, y, sample_weight)

#         return self

#     def predict(self, data, outcome=None, treatment=None):
#         if treatment is not None:
#             data = data.drop(treatment, axis=1)
#         if outcome is not None:
#             data = data.drop(outcome, axis=1)

#         check_is_fitted(self)
#         proba = self.tree_.predict(data)
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
#         return self.tree_.apply(X)

#     @property
#     def feature_importance(self):
#         pass
