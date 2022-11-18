from sklearn.utils import check_random_state

from .tree import GrfTree
from ._base_forest import BaseCausalForest

# causal forest without local centering technique
class GRForest(BaseCausalForest):
    def __init__(
        self,
        n_estimators=100,
        *,
        sub_sample_num=None,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        n_jobs=None,
        random_state=None,
        # categories="auto",
        ccp_alpha=0.0,
        is_discrete_treatment=True,
        is_discrete_outcome=False,
        verbose=0,
        warm_start=False,
        honest_subsample_num=None,
    ):
        base_estimator = GrfTree()

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=(
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "random_state",
                "ccp_alpha",
                "honest",
            ),
            n_jobs=n_jobs,
            # categories=categories,
            random_state=random_state,
            sub_sample_num=sub_sample_num,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
            is_discrete_treatment=is_discrete_treatment,
            is_discrete_outcome=is_discrete_outcome,
            verbose=verbose,
            warm_start=warm_start,
            honest_subsample_num=honest_subsample_num,
        )
        """
        A GRForest for estimating the causal effect.
        
        Parameters
        ----------
        n_estimators : int, default=100 
            The number of trees for growing the GRF.

        sub_sample_num: int or float, default=None 
            The number of samples to train each individual tree.
            - If a float is given, then the number of ``sub_sample_num*n_samples`` samples will be 
                sampled to train a single tree
            - If an int is given, then the number of ``sub_sample_num`` samples will be sampled to 
                train a single tree

        max_depth: int, default=None
            The max depth that a single tree can reach. If ``None`` is given, then there is no limit of
            the depth of a single tree.
        
        min_samples_split: int, default=2 
            The minimum number of samples required to split an internal node:
            - If int, then consider `min_samples_split` as the minimum number.
            - If float, then `min_samples_split` is a fraction and
            `ceil(min_samples_split * n_samples)` are the minimum
            number of samples for each split.

        min_samples_leaf: int or float, default=1
            The minimum number of samples required to be at a leaf node.
            A split point at any depth will only be considered if it leaves at
            least ``min_samples_leaf`` training samples in each of the left and
            right branches.  This may have the effect of smoothing the model,
            especially in regression.
                
                - If int, then consider `min_samples_leaf` as the minimum number.
                - If float, then `min_samples_leaf` is a fraction and `ceil(min_samples_leaf * n_samples)`
                    are the minimum number of samples for each node.

        min_weight_fraction_leaf: float, default=0.0 
            The minimum weighted fraction of the sum total of weights (of all
            the input samples) required to be at a leaf node. Samples have
            equal weight when sample_weight is not provided.
        
        max_features: int, float or {"sqrt", "log2"}, default=None
            The number of features to consider when looking for the best split:
            
                - If int, then consider `max_features` features at each split.
                - If float, then `max_features` is a fraction and `int(max_features * n_features)` features are considered at each split.
                - If "sqrt", then `max_features=sqrt(n_features)`.
                - If "log2", then `max_features=log2(n_features)`.
                - If None, then `max_features=n_features`.

        random_state: int 
            Controls the randomness of the estimator.
        
        max_leaf_nodes: int, default=None 
            Grow a tree with ``max_leaf_nodes`` in best-first fashion.
            Best nodes are defined as relative reduction in impurity.
            If None then unlimited number of leaf nodes.

        min_impurity_decrease: float, default=0.0 
            A node will be split if this split induces a decrease of the impurity
            greater than or equal to this value.
        
        n_jobs: int, default=None 
            The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
            :meth:`decision_path` and :meth:`apply` are all parallelized over the
            trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
            context. ``-1`` means using all processors. See :term:`Glossary
            <n_jobs>` for more details.

        verbose: int, default=0 
            Controls the verbosity when fitting and predicting

        honest_subsample_num: int or float, default=None 
            The number of samples to train each individual tree in an honest manner. Typically 
            set this value will have better performance. Use all ``sub_sample_num`` if ``None`` 
            is given.
            - If a float is given, then the number of ``honest_subsample_num*sub_sample_num`` samples
                will be used to train a single tree while the rest ``(1 - honest_subsample_num)*sub_sample_num``
                samples will be used to label the trained tree.
            - If an int is given, then the number of ``honest_subsample_num`` samples will be sampled 
                to train a single tree while the rest ``sub_sample_num - honest_subsample_num`` samples will 
                be used to label the trained tree.
        """

    def fit(
        self,
        data,
        outcome,
        treatment,
        adjustment=None,
        covariate=None,
        sample_weight=None,
    ):
        """Fit the model on data to estimate the causal effect.

        Parameters
        ----------
        data: pandas.DataFrame
            The input samples for the est_model to estimate the causal effects
            and for the CEInterpreter to fit.
        outcome: list of str, optional
            Names of the outcomes.
        treatment: list of str, optional
            Names of the treatments.
        covariate: list of str, optional, default=None
            Names of the covariate vectors.
        adjustment: list of str, optional, default=None
            This will be the same as the covariate.
        sample_weight:  ndarray, optional, default=None
            Weight of each sample of the training set.

        Returns
        ----------
            An instance of GRForest.
        """
        adjustment = None
        super().fit(
            data,
            outcome,
            treatment,
            adjustment=adjustment,
            covariate=covariate,
            sample_weight=sample_weight,
        )
