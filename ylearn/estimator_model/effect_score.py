"""
Beside effect score which measures the ability of estimating the causal
effect, we should also implement training_score measuring
performances of machine learning models.
"""
# from collections import defaultdict

import inspect
from asyncio.log import logger
import numpy as np
import pandas as pd

# from sklearn import clone
# from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold

from .double_ml import DoubleML
from .utils import convert2array, get_wv, get_tr_ctrl
from ylearn.utils import logging

logger = logging.get_logger(__name__)


class RLoss(DoubleML):
    """
    Estimator models for estimating causal effects can not be easily evaluated
    dut to the fact that the true effects are not directly observed. This differs
    from the usual machine learning tasks.

    Authors in [1] proposed a framework, a schema suggested by [2], to evaluate causal
    effects estimated by different estimator models. Roughly speaking, this
    framework is a direct application of the double machine learning methods.
    Specifically, for a causal effect model ce_model (trained in a training set)
    that is waited to be evaluated, we
        1. Train a model y_model to estimate the outcome and a x_model to
            estimate the treatment in a validation set, which is usually not
            the same as the training set;
        2. In the validation set v, let y_res and x_res denote the differences
                y_res = y - y_pred(v),
                x_res = x - x_pred(v),
            and
                tau(v)
            denote the causal effects estimated by the ce_model in the validation
            set, then the metric of the causal effect of the ce_model is
            calculated as
                E_{V}[(y_res - x_res * tau)^2].

    Attributes
    ----------
    _is_fitted : bool
        True if the model is fitted ortherwise False.

    x_model : estimator
        Machine learning models for fitting x. Any such models should implement
        the fit and predict (also predict_proba if x is discrete) methods

    y_model : estimator
        Machine learning models for fitting y.

    yx_model : estimator
        Machine learning models for fitting the residual of y on residual of x.
        Currently this should be a linear regression model.

    adjustment_transformer : transformer
        Transformer for adjustment variables, by default None.

    covariate_transformer : transformer
        Transformer for covariate variables, by default None.

    is_discrete_treatment : bool

    categories : str or list

    random_state : int

    cf_fold : int
        The number of folds for performing cross fit, by default 1

    x_hat_dict : defaultdict(list)
        Cached values when fitting the treatment model.

    y_hat_dict : defaultdict(list)
        Cached values when fitting the outcome model.

    Methods
    ----------
    fit(data, outcome, treatment, adjustment, covariate)
        Fit the RLoss estimator model in the validation set.

    comp_transformer(x, categories='auto')
        Transform the discrete treatment into one-hot vectors.

    score(self, test_estimator)
        Calculate the metric value of the RLoss for the test_estimator in the
        validation set.

    Example
    ----------
        from sklearn.ensemble import RandomForestRegressor

        from ylearn.exp_dataset.exp_data import single_binary_treatment
        from ylearn.estimator_model.meta_learner import TLearner

        train, val, te = single_binary_treatment()
        rloss = RLoss(
            x_model=RandomForestClassifier(),
            y_model=RandomForestRegressor(),
            cf_fold=1,
            is_discrete_treatment=True
        )
        rloss.fit(
            data=val,
            outcome=outcome,
            treatment=treatment,
            adjustment=adjustment,
            covariate=covariate,
        )

        est = TLearner(model=RandomForestRegressor())
        est.fit(
            data=train,
            treatment=treatment,
            outcome=outcome,
            adjustment=adjustment,
            covariate=covariate,
        )

    >>> rloss.score(est)
    0.20451977

    Reference
    ----------
    [1] A. Schuler, et al. A comparison of methods for model selection when
    estimating individual treatment effects. arXiv:1804.05146.
    [2] X. Nie, et al. Quasi-Oracle estimation of heterogeneous treatment effects.
        arXiv: 1712.04912.
    """

    def __init__(
        self,
        x_model,
        y_model,
        cf_fold=1,
        adjustment_transformer=None,
        covariate_transformer=None,
        random_state=2022,
        is_discrete_treatment=False,
        categories="auto",
    ):
        """
        Parameters
        ----------
        x_model : estimator
            Machine learning models for fitting x. Any such models should implement
            the fit and predict (also predict_proba if x is discrete) methods

        y_model : estimator
            Machine learning models for fitting y.

        yx_model : estimator, optional
            Machine learning models for fitting the residual of y on residual of x.

        cf_fold : int, optional
            The number of folds for performing cross fit, by default 1

        adjustment_transformer : transformer, optional
            Transformer for adjustment variables, by default None

        covariate_transformer : transformer, optional
            Transformer for covariate variables, by default None

        random_state : int, optional
            Random seed, by default 2022

        is_discrete_treatment : bool, optional
            If the treatment variables are discrete, set this as True, by default False

        categories : str, optional
        """
        self.yx_model = None

        super().__init__(
            x_model=x_model,
            y_model=y_model,
            cf_fold=cf_fold,
            adjustment_transformer=adjustment_transformer,
            covariate_transformer=covariate_transformer,
            random_state=random_state,
            is_discrete_treatment=is_discrete_treatment,
            categories=categories,
        )

    def fit(
        self,
        data,
        outcome,
        treatment,
        adjustment=None,
        covariate=None,
        combined_treatment=True,
    ):
        """Fit the RLoss estimator model. Note that the trainig of a DML has two stages, where we implement them in
        :py:func:`_fit_1st_stage` and :py:func:`_fit_2nd_stage`.

        Parameters
        ----------
        data : pandas.DataFrame
            The dataset used for training the model

        outcome : str or list of str, optional
            Names of the outcome variables

        treatment : str or list of str
            Names of the treatment variables

        adjustment : str or list of str, optional
            Names of the adjustment variables, by default None

        covariate : str or list of str, optional
            Names of the covariate variables, by default None

        combined_treatment : bool
            When combined_treatment is set to True, then if there are multiple
            treatments, we can use the combined_treatment technique to covert
            the multiple discrete classification tasks into a single discrete
            classification task. For an example, if there are two different
            binary treatments:
                treatment_1: x_1 | x_1 \in {'sleep', 'run'},
                treatment_2: x_2 | x_2 \in {'study', 'work'},
            then we can convert to these two binary classification tasks into
            a single classification with 4 different classes:
                treatment: x | x \in {0, 1, 2, 3},
            where, for example, 1 stands for ('sleep' and 'stuy').

        Returns
        ----------
        instance of RLoss
            The fitted RLoss model for evaluating other estimator models in
            the validation set.
        """
        assert covariate is not None, "Need covariates to use RLoss."

        self.combined_treatment = combined_treatment

        # super().fit(
        #     data, outcome, treatment,
        #     adjustment=adjustment,
        #     covariate=covariate,
        # )
        self.outcome = outcome
        self.treatment = treatment
        self.adjustment = adjustment
        self.covariate = covariate

        y, x, w, v = convert2array(data, outcome, treatment, adjustment, covariate)
        self._w = w
        self._v = v
        self._y_d = y.shape[1]
        cfold = self.cf_fold

        if self.adjustment_transformer is not None and w is not None:
            w = self.adjustment_transformer.fit_transform(w)

        if self.covariate_transformer is not None and v is not None:
            v = self.covariate_transformer.fit_transform(v)

        if self.is_discrete_treatment:
            if self.categories == "auto" or self.categories is None:
                categories = "auto"
            else:
                categories = list(self.categories)

            # convert discrete treatment features to onehot vectors
            x = self.comp_transormer(x, categories)

        self._x_d = x.shape[1]
        wv = get_wv(w, v)

        # step 1: split the data
        if cfold > 1:
            cfold = int(cfold)
            folds = [KFold(n_splits=cfold).split(x), KFold(n_splits=cfold).split(y)]
        else:
            folds = None

        # step 2: cross fit to give the estimated y and x
        self.x_hat_dict, self.y_hat_dict = super()._fit_1st_stage(
            self.x_model, self.y_model, y, x, wv, folds=folds
        )
        x_hat = self.x_hat_dict["paras"][0].reshape((x.shape))
        y_hat = self.y_hat_dict["paras"][0].reshape((y.shape))

        x_prime = x - x_hat
        y_prime = y - y_hat

        self.x_hat_dict["res"].append(x_prime)
        self.y_hat_dict["res"].append(y_prime)

        self._is_fitted = True

        return self

    def score(self, test_estimator, treat=None, control=None):
        """Calculate the RLoss as a metric of the causal effect estimated by
        the test_estimator.

        Parameters
        ----------
        test_estimator : BaseEstModel
            Any fitted estimator model for causal effect.

        treat : float or ndarray, optional
            In the case of single discrete treatment, treat should be an int or
            str of one of all possible treatment values which indicates the
            value of the intended treatment;
            in the case of multiple discrete treatment, treat should be a list
            or an ndarray where treat[i] indicates the value of the i-th intended
            treatment, for example, when there are multiple
            discrete treatments, array(['run', 'read']) means the treat value of
            the first treatment is taken as 'run' and that of the second treatment
            is taken as 'read';
            in the case of continuous treatment, treat should be a float or a
            ndarray, by default None

        control : float or ndarray, optional
            This is similar to the cases of treat, by default None

        Returns
        -------
        float
            The score for the test_estimator
        """
        x_prime, y_prime = self.x_hat_dict["res"][0], self.y_hat_dict["res"][0]
        v = self._v
        w = self._w

        data_dict = {}
        if self.covariate is not None:
            for i, name in enumerate(test_estimator.covariate):
                data_dict[name] = v[:, i]
        if self.adjustment is not None:
            for i, name in enumerate(test_estimator.adjustment):
                data_dict[name] = w[:, i]

        test_data = pd.DataFrame(data_dict)

        # shape (n, y_d, x_d)
        est_params = inspect.signature(test_estimator.estimate).parameters.keys()
        est_options = {}
        dis_tr = test_estimator.is_discrete_treatment

        if "treat" in est_params and treat is not None:
            if hasattr(test_estimator, "comp_transformer"):
                treat = get_tr_ctrl(
                    treat,
                    self.comp_transormer,
                    treat=True,
                    one_hot=False,
                    discrete_treat=dis_tr,
                )
            est_options["treat"] = treat
        if "control" in est_params and control is not None:
            if hasattr(test_estimator, "comp_transformer"):
                control = get_tr_ctrl(
                    control,
                    self.comp_transormer,
                    treat=False,
                    one_hot=False,
                    discrete_treat=dis_tr,
                )
            est_options["control"] = control
        test_effect = test_estimator.estimate(data=test_data, **est_options)
        logger.info(
            f"Calculating the score: {test_estimator.__repr__()} finished estimating."
        )
        if self.is_discrete_treatment and self.combined_treatment:
            logger.info("using combined treat technique for discrete treatment.")
            x_prime = x_prime[:, test_estimator.treat].reshape(-1, 1)
            test_effect = test_effect.reshape(v.shape[0], self._y_d, 1)

        y_pred = np.einsum("nij, nj->ni", test_effect, x_prime)
        rloss = np.mean((y_prime - y_pred) ** 2, axis=0)

        return rloss

    def _fit_1st_stage(self, x_model, y_model, y, x, wv, folds=None, **kwargs):
        """Fit the models in the first stage.

        Parameters
        ----------
        x_model : estimator
            Any x_model should have the fit and predict (also predict_proba if
            x is discrete) methods.

        y_model : estimator
            Any y_model should have the fit and predict (also predict_proba if
            y is discrete) methods.

        y : ndarray, shape (n, y_d)
            The outcome vector

        x : ndarray, shape (n, x_d)
            The treatment vector

        wv : ndarray, shape (n, w_d + v_d)
            The covariate and adjustment vector

        folds : sklearn.model_selection.KFold, optional

        Returns
        -------
        tuple of dict
            The first dict containes the values of the fitted y_model and the
            second dict the fitted x_model
        """
        if self._x_d == 1:
            x = np.ravel(x)
        if self._y_d == 1:
            y = np.ravel(y)

        if folds is not None:
            x_folds, y_folds = folds
        else:
            x_folds, y_folds = None, None

        logger.info(f"_fit_1st_stage: fitting x_model {type(x_model).__name__}")
        x_hat_dict = self._cross_fit(
            x_model, wv, target=x, folds=x_folds, is_ymodel=False, **kwargs
        )

        logger.info(f"_fit_1st_stage: fitting y_model {type(y_model).__name__}")
        y_hat_dict = self._cross_fit(
            y_model, wv, target=y, folds=y_folds, is_ymodel=True, **kwargs
        )
        return (x_hat_dict, y_hat_dict)

    def __repr__(self) -> str:
        return f"RLoss"


class PredLoss:
    def __init__(self) -> None:
        pass

    def fit(
        self,
    ):
        pass

    def score(self):
        pass


# class Score:
#     def __init__(self):
#         pass

#     def fit(self, data):
#         pass

#     def score(self,):
#         pass
