import pandas as pd
import numpy as np

from copy import deepcopy
from collections import defaultdict

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from .base_models import BaseEstLearner
from estimator_model.utils import convert2array, get_group_ids
np.random.seed(2022)


class SLearner(BaseEstLearner):
    """
    SLearn uses one machine learning model to compute the causal effects.
    Specifically, we fit a model to predict outcome (y) from treatment (x) and
    adjustment (w):
        y = f(x, w)
    and the causal effect is
        causal_effect = f(x=1, w) - f(x=0, w).

    Attributes
    ----------
    ml_model_dic : dict
        A dictionary of default machine learning sklearn models currently
        including
            'LR': LinearRegression
            'LogistR': LogisticRegression.
    ml_model : str, optional
        The machine learning model for modeling the relation between outcome
        and (treatment, adjustment).

    Methods
    ----------
    _prepare4est(data, outcome, treatment, adjustment, individual=None)
        Prepare (fit the model) for estimating various quantities including
        ATE, CATE, ITE, and CITE.
    estimate(data, outcome, treatment, adjustment, quantity='ATE',
                 condition_set=None, condition=None, individual=None)
        Integrate estimations for various quantities into a single method.
    estimate_ate(self, data, outcome, treatment, adjustment)
    estimate_cate(self, data, outcome, treatment, adjustment,
                      condition_set, condition)
    estimate_ite(self, data, outcome, treatment, adjustment, individual)
    estimate_cite(self, data, outcome, treatment, adjustment,
                      condition_set, condition, individual)
    """

    def __init__(
        self,
        model,
        random_state=2022,
        categories='auto',
        *args,
        **kwargs
    ):
        """
        Parameters
        ----------
        ml_model : str, optional
            If str, ml_model is the name of the machine learning mdoels used
            for our TLearner. If not str, then ml_model should be some valid
            machine learning model wrapped by the class MLModels.
        """
        self.model = model

        super().__init__(
            random_state=random_state,
            categories=categories,
            *args,
            **kwargs,
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
        assert adjustment is not None or covariate is not None, \
            'Need adjustment set or covariates to perform estimation.'

        self._is_fitted = True

        n = len(data)
        self.outcome = outcome
        self.treatment = treatment
        self.adjustment = adjustment
        self.covariate = covariate

        y, x, w, v = convert2array(
            data, outcome, treatment, adjustment, covariate
        )

        self.y_d = y.shape[1]

        if self.categories == 'auto' or self.categories is None:
            categories = 'auto'
        else:
            categories = list(self.categories)

        # self.transformer = OneHotEncoder(categories=categories)
        self.transformer = OrdinalEncoder(categories=categories)
        self.transformer.fit(x)
        x = self.transformer.transform(x).toarray()

        # For multiple treatments: divide the data into several groups
        group_categories = self.transformer.categories_
        num_treatments = len(group_categories)

        if treat is not None:
            if not isinstance(treat, int):
                assert len(treat) == num_treatments
            treat = np.repeat(
                np.array(list(treat)).reshape(1, -1), n, axis=0
            )
        else:
            treat = np.ones((n, num_treatments))

        if control is not None:
            if not isinstance(control, int):
                assert len(control) == num_treatments
            control = np.repeat(
                np.array(list(control)).reshape(1, -1), n, axis=0
            )
        else:
            control = np.zeros((n, num_treatments))

        self.treat_index = treat[0]
        self.control_index = control[0]

        if w is None:
            wv = v
        else:
            if v is not None:
                wv = np.concatenate((w, v), axis=1)
            else:
                wv = w

        self._wv = wv
        x = np.concatenate((wv, x), axis=1)
        self.model.fit(x, y)

        return self

    def _prepare4est(self, data=None, *args, **kwargs):
        if not hasattr(self, 'is_fitted'):
            raise Exception('The estimator has not been fitted yet.')

        if data is None:
            wv = self._wv
        else:
            w, v = convert2array(
                data, self.adjustment, self.covariate
            )
            if w is None:
                wv = v
            else:
                if v is not None:
                    wv = np.concatenate((w, v), axis=1)
                else:
                    wv = w

        n = wv.shape[0]
        xt = np.repeat(self.treat_index, n, axis=0)
        xt = np.concatenate((wv, xt), axis=1)
        x0 = np.repeat(self.control_index, n, axis=0)
        x0 = np.concatenate((wv, x0), axis=1)

        yt = self.model.predict(xt)
        y0 = self.model.predict(x0)

        return yt, y0

    def estimate(
        self,
        data=None,
        quantity='CATE',
        *args,
        **kwargs
    ):
        yt, y0 = self._prepare4est(data, *args, **kwargs)
        return np.mean(yt-y0, axis=0)


class TLearner(BaseEstLearner):
    """
    TLearner uses two machine learning models to estimate the causal
    effect. Specifically, we
    1. fit two models for the treatment group (x=1) and control group (x=0),
        respectively:
        y1 = x1_model(w) with data where x=1,
        y0 = x0_model(w) with data where x=0;
    2. compute the causal effect as the difference between these two models:
        causal_effect = x1_model(w) - x0_model(w).

    Attributes
    -----------
    ml_model_dic : dict
        A dictionary of default machine learning sklearn models currently
        including
            'LR': LinearRegression
            'LogistR': LogisticRegression.
    x1_model : MLModel, optional
        The machine learning model for the treatment group data.
    x0_model : MLModel, optional
        The machine learning model for the control group data.

    Methods
    ----------
    _prepare4est(data, outcome, treatment, adjustment, individual=None)
        Prepare (fit the model) for estimating various quantities including
        ATE, CATE, ITE, and CITE.
    estimate(data, outcome, treatment, adjustment, quantity='ATE',
                 condition_set=None, condition=None, individual=None)
        Integrate estimations for various quantities into a single method.
    estimate_ate(self, data, outcome, treatment, adjustment)
    estimate_cate(self, data, outcome, treatment, adjustment,
                      condition_set, condition)
    estimate_ite(self, data, outcome, treatment, adjustment, individual)
    estimate_cite(self, data, outcome, treatment, adjustment,
                      condition_set, condition, individual)
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        ml_model : str, optional
            If str, ml_model is the name of the machine learning mdoels used
            for our TLearner. If not str, then ml_model should be some valid
            machine learning model wrapped by the class MLModels.
        """
        super().__init__()
        model = kwargs['ml_model']

        if type(model) is str:
            model = self.ml_model_dic[model]

        self.x1_model = model
        self.x0_model = deepcopy(model)

    def _prepare4est(self, data, outcome, treatment, adjustment, individual=None):
        r"""Prepare (fit the model) for estimating the quantities
            ATE: E[y|do(x_1)] - E[y|do(x_0)] = E_w[E[y|x=x_1,w] - E[y|x=x_0, w]
                                           := E_{adjustment}[
                                               Delta E[outcome|treatment,
                                                                adjustment]]
            CATE: E[y|do(x_1), z] - E[y|do(x_0), z] = E_w[E[y|x=x_1, w, z] -
                                                        E[y|x=x_0, w, z]]
            ITE: y_i(do(x_1)) - y_i(do(x_0))
            CITE: y_i(do(x_1))|z_i - y_i(do(x_0))|z_i

        Parameters
        ----------
        data : DataFrame
        outcome : string
            Name of the outcome.
        treatment : string
            Name of the treatment.
        adjustment : set or list
            The adjutment set for the causal effect,
            i.e., P(outcome|do(treatment)) =
                \sum_{adjustment} P(outcome|treatment, adjustment)P(adjustment)
        individual : DataFrame, default to None
            The individual data for computing its causal effect.

        Returns
        ----------
        np.array
        """
        data_without_treatment = data.drop([treatment], axis=1)
        x1_data = data_without_treatment.loc[data[treatment] > 0]
        x0_data = data_without_treatment.loc[data[treatment] <= 0]
        self.x1_model.fit(x1_data[adjustment], x1_data[outcome])
        self.x0_model.fit(x0_data[adjustment], x0_data[outcome])

        if individual:
            data_ = individual[adjustment]
        else:
            data_ = data[adjustment]

        result = (
            self.x1_model.predict(data_) - self.x0_model.predict(data_)
        )
        return result


class XLearner(BaseEstLearner):
    """
    The XLearner is composed of 3 steps:
        1. Train two different models for the control group and treated group
            f_0(w), f_1(w)
        2. Generate two new datasets (h_0, w) using the control group and
            (h_1, w) using the treated group where
            h_0 = f_1(w) - y_0(w), h_1 = y_1(w) - f_0(w). Then train two models
            k_0(w) and k_1(w) in these datasets.
        3. Get the final model using the above two models:
            g(w) = k_0(w)a(w) + k_1(w)(1 - a(w)).
    Finally,  we estimate the ATE as follows:
        ATE = E_w(g(w)).
    See Kunzel, et al., (https://arxiv.org/abs/1706.03461) for reference.

    Attributes
    ----------
    ml_model_dic : dict
        A dictionary of default machine learning sklearn models currently
        including
            'LR': LinearRegression
            'LogistR': LogisticRegression.
    f1 : MLModel, optional
        Machine learning model for the treatment gropu in the step 1.
    f0 : MLModel, optional
        Machine learning model for the control gropu in the step 1.
    k1 : MLModel, optional
        Machine learning model for the treatment gropu in the step 2.
    k0 : MLModel, optional
        Machine learning model for the control gropu in the step 2.

    Methods
    ----------
    _prepare4est(data, outcome, treatment, adjustment, individual=None)
        Prepare (fit the model) for estimating various quantities including
        ATE, CATE, ITE, and CITE.
    estimate(data, outcome, treatment, adjustment, quantity='ATE',
                 condition_set=None, condition=None, individual=None)
        Integrate estimations for various quantities into a single method.
    estimate_ate(self, data, outcome, treatment, adjustment)
    estimate_cate(self, data, outcome, treatment, adjustment,
                      condition_set, condition)
    estimate_ite(self, data, outcome, treatment, adjustment, individual)
    estimate_cite(self, data, outcome, treatment, adjustment,
                      condition_set, condition, individual)
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        ml_model : str, optional
            If str, ml_model is the name of the machine learning mdoels used
            for our TLearner. If not str, then ml_model should be some valid
            machine learning model wrapped by the class MLModels.
        """
        super().__init__()
        model = kwargs['ml_model']

        if type(model) is str:
            model = self.ml_model_dic[model]

        self.f1 = model
        self.f0 = deepcopy(model)
        self.k1 = deepcopy(model)
        self.k0 = deepcopy(model)

    def _prepare4est(self, data, outcome, treatment, adjustment, individual=None):
        r"""Prepare (fit the model) for estimating the quantities
            ATE: E[y|do(x_1)] - E[y|do(x_0)] = E_w[E[y|x=x_1,w] - E[y|x=x_0, w]
                                           := E_{adjustment}[
                                               Delta E[outcome|treatment,
                                                                adjustment]]
            CATE: E[y|do(x_1), z] - E[y|do(x_0), z] = E_w[E[y|x=x_1, w, z] -
                                                        E[y|x=x_0, w, z]]
            ITE: y_i(do(x_1)) - y_i(do(x_0))
            CITE: y_i(do(x_1))|z_i - y_i(do(x_0))|z_i

        Parameters
        ----------
        data : DataFrame
        outcome : string
            Name of the outcome.
        treatment : string
            Name of the treatment.
        adjustment : set or list
            The adjutment set for the causal effect,
            i.e., P(outcome|do(treatment)) =
                \sum_{adjustment} P(outcome|treatment, adjustment)P(adjustment)
        individual : DataFrame, default to None
            The individual data for computing its causal effect.
        """
        # step 1
        data_without_treatment = data.drop([treatment], axis=1)
        x1_data = data_without_treatment.loc[data[treatment] > 0]
        x0_data = data_without_treatment.loc[data[treatment] <= 0]
        self.f1.fit(x1_data[adjustment], x1_data[outcome])
        self.f0.fit(x0_data[adjustment], x0_data[outcome])

        # setp 2
        h1_data = x1_data.drop(outcome, axis=1)
        h0_data = x0_data.drop(outcome, axis=1)
        h1 = x1_data[outcome] - self.f0.predict(h1_data[adjustment])
        h0 = self.f1.predict(h0_data[adjustment]) - x0_data[outcome]
        self.k1.fit(h1_data[adjustment], h1)
        self.k0.fit(h0_data[adjustment], h0)

        # step 3
        if individual:
            data_ = individual[adjustment]
        else:
            data_ = data[adjustment]
        # TODO: more choices of rho
        rho = 0.5
        result = rho * self.k1.predict(data_) + \
            (1 - rho) * self.k0.predict(data_)
        return result


# class DragonNet(MetaLearner):
#     """
#     See Shi., et al., (https://arxiv.org/pdf/1906.02120.pdf) for reference.

#     Args:
#         MetaLearner ([type]): [description]
#     """

#     def __init__(self) -> None:
#         super().__init__()
