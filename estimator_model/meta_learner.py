
from ast import arg
from re import S
import re
import numpy as np

from sklearn import clone
from sklearn.preprocessing import OrdinalEncoder

from .base_models import BaseEstLearner
from estimator_model.utils import (convert2array, get_group_ids,
                                   get_treat_control)


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
        self._is_fitted = False

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
        combined_treatment=True,
        *args,
        **kwargs
    ):
        assert adjustment is not None or covariate is not None, \
            'Need adjustment set or covariates to perform estimation.'

        n = len(data)
        self.outcome = outcome
        self.treatment = treatment
        self.adjustment = adjustment
        self.covariate = covariate
        self.combined_treat = combined_treatment

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
        x = self.transformer.transform(x)

        # For multiple treatments: divide the data into several groups
        group_categories = self.transformer.categories_
        num_treatments = len(group_categories)

        treat = get_treat_control(treat, n, num_treatments, True)
        control = get_treat_control(control, n, num_treatments, False)
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

        if y.shape[1] == 1:
            y = y.ravel()

        self.model.fit(x, y, *args, **kwargs)

        self._is_fitted = True

        return self

    def _prepare4est(self, data=None, *args, **kwargs):
        if not self._is_fitted:
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
        x0 = np.repeat(self.control_index, n, axis=0)
        if len(xt.shape) == 1:
            xt = xt.reshape(-1, 1)
            x0 = x0.reshape(-1, 1)
        xt = np.concatenate((wv, xt), axis=1)
        x0 = np.concatenate((wv, x0), axis=1)

        yt = self.model.predict(xt)
        y0 = self.model.predict(x0)

        return yt, y0

    def estimate(
        self,
        data=None,
        quantity=None,
        *args,
        **kwargs
    ):
        yt, y0 = self._prepare4est(data, *args, **kwargs)
        if quantity == 'CATE' or quantity == 'ATE':
            return np.mean(yt - y0, axis=0)
        else:
            return yt - y0


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
        self.xt_model = clone(model)
        self.x0_model = clone(model)
        self._is_fitted = False

        super().__init__(
            random_state=random_state,
            categories=categories,
            *args,
            **kwargs
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
        combined_treatment=True,
        *args,
        **kwargs
    ):
        assert adjustment is not None or covariate is not None, \
            'Need adjustment set or covariates to perform estimation.'

        self._n = len(data)
        self.outcome = outcome
        self.treatment = treatment
        self.adjustment = adjustment
        self.covariate = covariate
        self.combined_treat = combined_treatment

        y, x, w, v = convert2array(
            data, outcome, treatment, adjustment, covariate
        )

        if self.categories == 'auto' or self.categories is None:
            categories = 'auto'
        else:
            categories = list(self.categories)

        self.transformer = OrdinalEncoder(categories=categories)
        self.transformer.fit(x)
        x = self.transformer.transform(x)

        group_categories = self.transformer.categories_
        num_treatments = len(group_categories)

        if w is None:
            wv = v
        else:
            if v is not None:
                wv = np.concatenate((w, v), axis=1)
            else:
                wv = w

        self._wv = wv
        if combined_treatment:
            return self._fit_combined_treat(
                x, wv, y, treat, control, num_treatments, *args, **kwargs
            )
        else:
            return self._fit_separate_treat()

    def estimate(
        self,
        data=None,
        quantity=None,
        *args,
        **kwargs
    ):
        yt, y0 = self._prepare4est(data, *args, **kwargs)
        if quantity == 'CATE' or quantity == 'ATE':
            return np.mean(yt-y0, axis=0)
        else:
            return yt - y0

    def _prepare4est(self, data=None, *args, **kwargs):
        if not self._is_fitted:
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

        if wv.shape[1] == 1:
            wv = wv.ravel()

        yt = self.xt_model.predict(wv)
        y0 = self.x0_model.predict(wv)

        return yt, y0

    def _fit_combined_treat(
        self,
        x, wv, y, n,
        treat,
        control,
        num_treatments,
        *args,
        **kwargs
    ):
        treat = get_treat_control(treat, n, num_treatments, True)
        control = get_treat_control(control, n, num_treatments, False)
        self.treat_index = treat[0]
        self.control_index = control[0]

        wv_treat, y_treat = get_group_ids(treat, x, wv, y)
        wv_control, y_control = get_group_ids(control, x, wv, y)

        if wv_treat.shape[1] == 1:
            wv_treat = wv_treat.ravel()
            wv_control = wv_control.ravel()
        if y_treat.shape[1] == 1:
            y_treat = y_treat.ravel()
            y_control = y_control.ravel()

        self.xt_model.fit(wv_treat, y_treat, *args, **kwargs)
        self.x0_model.fit(wv_control, y_control, *args, **kwargs)

        self._is_fitted = True

        return self

    def _fit_separate_treat(self):
        pass

    def _prepare_combined_treat(self):
        pass

    def _prepare_combined_control(self):
        pass


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
        self.ft_model = clone(model)
        self.f0_model = clone(model)
        self.kt_model = clone(model)
        self.k0_model = clone(model)
        self._is_fitted = False

        super().__init__(
            random_state=random_state,
            categories=categories,
            *args,
            **kwargs
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
        combined_treatment=True,
        *args,
        **kwargs,
    ):
        assert adjustment is not None or covariate is not None, \
            'Need adjustment set or covariates to perform estimation.'

        self._n = len(data)
        self.outcome = outcome
        self.treatment = treatment
        self.adjustment = adjustment
        self.covariate = covariate
        self.combined_treat = combined_treatment

        y, x, w, v = convert2array(
            data, outcome, treatment, adjustment, covariate
        )

        if self.categories == 'auto' or self.categories is None:
            categories = 'auto'
        else:
            categories = list(self.categories)

        self.transformer = OrdinalEncoder(categories=categories)
        self.transformer.fit(x)
        x = self.transformer.transform(x).toarray()

        group_categories = self.transformer.categories_
        num_treatments = len(group_categories)

        if w is None:
            wv = v
        else:
            if v is not None:
                wv = np.concatenate((w, v), axis=1)
            else:
                wv = w

        if combined_treatment:
            n = self._n
            return self._fit_combined_treat(
                x, wv, y, n, treat, control, num_treatments, *args, **kwargs
            )
        else:
            return self._fit_separate_treat()

    def _prepare4est(self, data=None, *args, **kwargs):
        if not self._is_fitted:
            raise Exception('The estimator has not been fitted yet.')

        if self.combined_treat:
            kt_pred, k0_pred = self._prepare_combined_treat(
                data, *args, **kwargs
            )
        else:
            kt_pred, k0_pred = self._prepare_separate_treat(
                data, *args, **kwargs
            )

        return kt_pred, k0_pred

    def estimate(
        self,
        data=None,
        rho=0.5,
        quantity='CATE',
        *args,
        **kwargs
    ):
        kt_pred, k0_pred = self._prepare4est(data, *args, **kwargs)
        pred = rho * kt_pred + (1 - rho) * k0_pred
        if quantity == 'CATE' or quantity == 'ATE':
            return np.mean(pred, axis=0)
        else:
            return pred

    def _fit_combined_treat(
        self,
        x, wv, y, n,
        treat,
        control,
        num_treatments,
        *args,
        **kwargs
    ):
        treat = get_treat_control(treat, n, num_treatments, True)
        control = get_treat_control(control, n, num_treatments, False)
        self.treat_index = treat[0]
        self.control_index = control[0]

        # Step 1
        # TODO: modify the generation of group ids for efficiency
        wv_treat, y_treat = get_group_ids(treat, x, wv, y)
        wv_control, y_control = get_group_ids(control, x, wv, y)

        if wv_treat.shape[1] == 1:
            wv_treat = wv_treat.ravel()
            wv_control = wv_control.ravel()
        if y_treat.shape[1] == 1:
            y_treat = y_treat.ravel()
            y_control = y_control.ravel()

        self.ft_model.fit(wv_treat, y_treat, *args, **kwargs)
        self.f0_model.fit(wv_control, y_control, *args, **kwargs)

        # Step 2
        h_treat_target = y_treat - self.f0_model.predict(wv_treat)
        h_control_target = self.ft_model.predict(wv_control) - y_control
        self.kt_model.fit(wv_treat, h_treat_target)
        self.k0_model.fit(wv_control, h_control_target)

        # Step 3
        # See _prepare4est
        self._wv_treat = wv_treat
        self._wv_control = wv_control

        self._is_fitted = True

        return self

    def _fit_separate_treat(self):
        pass

    def _prepare_combined_treat(self, data, *args, **kwargs):
        if data is None:
            wv_treat = self._wv_treat
            wv_control = self._wv_control
        else:
            n = self.n
            x, w, v = convert2array(
                data, self.treatment, self.adjustment, self.covariate
            )

            if w is None:
                wv = v
            else:
                if v is not None:
                    wv = np.concatenate((w, v), axis=1)
                else:
                    wv = w

            treat = np.repeat(self.treat_index, n, axis=0)
            control = np.repeat(self.control_index, n, axis=0)
            wv_treat = get_group_ids(treat, x, wv)
            wv_control = get_group_ids(control, x, wv)
            if wv_treat.shape[1] == 1:
                wv_treat = wv_treat.ravel()
                wv_control = wv_control.ravel()

        kt_pred = self.kt_model.predict(wv_treat)
        k0_pred = self.k0_model.predict(wv_control)

        return kt_pred, k0_pred

    def _prepare_separate_treat(self, data, *args, **kwargs):
        pass
