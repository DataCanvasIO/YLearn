from sklearn.tree import DecisionTreeRegressor

from ylearn.estimator_model.utils import convert2array

class CEInterpreter:
    def __init__(
        self, *,
        criterion='squared_error',
        splitter='best',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.,
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
        """
        self._is_fitted = False
        self.treatment = None
        self.outcome = None
        
        self._tree = DecisionTreeRegressor(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha
        )

    def fit(
        self,
        data,
        est_model,
        **kwargs
    ):
        assert est_model._is_fitted

        covariate = est_model.covariate
        assert covariate is not None, 'Need covariate to interpret the causal effect.'
        
        v = convert2array(data, covariate)[0]
        n = v.shape[0]
        self.v = v

        causal_effect = est_model.estimate(data=data, quantity=None, **kwargs)
        
        self._tree.fit(v, causal_effect.reshape((n, -1)))
        
        self._is_fitted = True

    def interpret(self):
        assert self._is_fitted, 'The model is not fitted yet. Please use the fit method first.'
        
        raise NotImplemented()