import numpy as np

from sklearn import clone
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from collections import defaultdict

from .base_models import BaseEstLearner
from estimator_model.utils import (convert2array, get_groups,
                                   get_treat_control, get_wv, cartesian)


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
        **kwargs
    ):
        assert adjustment is not None or covariate is not None, \
            'Need adjustment set or covariates to perform estimation.'

        self.outcome = outcome
        self.treatment = treatment
        self.adjustment = adjustment
        self.covariate = covariate
        self.combined_treat = combined_treatment

        y, x, w, v = convert2array(
            data, outcome, treatment, adjustment, covariate
        )

        self._y_d = y.shape[1]
        if self.categories == 'auto' or self.categories is None:
            categories = 'auto'
        else:
            categories = list(self.categories)

        # self.transformer = OneHotEncoder(categories=categories)
        wv = get_wv(w, v)
        self._wv = wv

        if combined_treatment:
            return self._fit_combined_treat(
                x, wv, y, treat, control, categories, **kwargs
            )
        else:
            return self._fit_separate_treat(
                x, wv, y, categories, **kwargs
            )

    def _prepare4est(self, data=None, **kwargs):
        if not self._is_fitted:
            raise Exception('The estimator has not been fitted yet.')

        if data is None:
            wv = self._wv
        else:
            w, v = convert2array(
                data, self.adjustment, self.covariate
            )
            wv = get_wv(w, v)

        if self.combined_treat:
            return self._prepare_combined_treat(wv)
        else:
            return self._prepare_separate_treat(wv)

    def estimate(
        self,
        data=None,
        quantity=None,
        **kwargs
    ):
        effect = self._prepare4est(data, **kwargs)
        if quantity == 'CATE' or quantity == 'ATE':
            return np.mean(effect, axis=0)
        else:
            return effect

    def _fit_combined_treat(
        self,
        x, wv, y,
        treat,
        control,
        categories,
        **kwargs
    ):
        self.transformer = OrdinalEncoder(categories=categories)
        self.transformer.fit(x)
        x = self.transformer.transform(x)

        # For multiple treatments: divide the data into several groups
        group_categories = self.transformer.categories_
        n_treatments = len(group_categories)

        treat = get_treat_control(treat, n_treatments, True)
        control = get_treat_control(control, n_treatments, False)
        self.treat = treat
        self.control = control

        x = np.concatenate((wv, x), axis=1)
        y = y.squeeze()

        self.model.fit(x, y, **kwargs)

        self._is_fitted = True

        return self

    def _fit_separate_treat(
        self,
        x, wv, y,
        categories,
        **kwargs
    ):
        self.transformer = OneHotEncoder(categories=categories)
        self.transformer.fit(x)
        x = self.transformer.transform(x).toarray()
        self._x_d = x.shape[1]
        x = np.concatenate((wv, x), axis=1)
        y = y.squeeze()

        # self.model.fit(x, y, *args, **kwargs)
        self.model.fit(x, y, **kwargs)

        self._is_fitted = True

        return self

    def _prepare_combined_treat(self, wv):
        n = wv.shape[0]
        xt = np.repeat(self.treat, n, axis=0)
        x0 = np.repeat(self.control, n, axis=0)
        if len(xt.shape) == 1:
            xt = xt.reshape(-1, 1)
            x0 = x0.reshape(-1, 1)
        xt = np.concatenate((wv, xt), axis=1)
        x0 = np.concatenate((wv, x0), axis=1)

        yt = self.model.predict(xt)
        y0 = self.model.predict(x0)
        return yt - y0

    def _prepare_separate_treat(self, wv):
        n = wv.shape[0]
        x_control = np.zeros((1, self._x_d))
        x_control[:, 0] = 1
        x_control = np.repeat(x_control, n, axis=0).astype(int)
        x_control = np.concatenate((wv, x_control), axis=1)

        f_nji = np.full((n, self._y_d, self._x_d - 1), np.NaN)
        f_nj0 = self.model.predict(x_control)

        for i in range(self._x_d - 1):
            x_treat = np.zeros((1, self._x_d))
            x_treat[:, i+1] = 1
            x_treat = np.repeat(x_treat, n, axis=0).astype(int)
            x_treat = np.concatenate((wv, x_treat), axis=1)
            fnji = (self.model.predict(x_treat) - f_nj0).reshape(n, self._y_d)
            f_nji[:, :, i] = fnji

        return f_nji.squeeze()


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
        **kwargs
    ):
        assert adjustment is not None or covariate is not None, \
            'Need adjustment set or covariates to perform estimation.'

        self.outcome = outcome
        self.treatment = treatment
        self.adjustment = adjustment
        self.covariate = covariate
        self.combined_treat = combined_treatment

        y, x, w, v = convert2array(
            data, outcome, treatment, adjustment, covariate
        )
        self._y_d = y.shape[1]

        if self.categories == 'auto' or self.categories is None:
            categories = 'auto'
        else:
            categories = list(self.categories)

        self.transformer = OrdinalEncoder(categories=categories)
        self.transformer.fit(x)
        x = self.transformer.transform(x)

        group_categories = self.transformer.categories_
        wv = get_wv(w, v)

        if combined_treatment:
            n_treatments = len(group_categories)
            return self._fit_combined_treat(
<<<<<<< HEAD
                x, wv, y, treat, control, n_treatments, *args, **kwargs
=======
                x, wv, y, treat, control, num_treatments, **kwargs
>>>>>>> 9746a0e64a7b2ced2e52dce5d74c5c4992567ff9
            )
        else:
            return self._fit_separate_treat(
                x, wv, y, group_categories, **kwargs
            )

    def estimate(
        self,
        data=None,
        quantity=None,
        **kwargs
    ):
        effect = self._prepare4est(data)
        if quantity == 'CATE' or quantity == 'ATE':
            return np.mean(effect, axis=0)
        else:
            return effect

    def _prepare4est(self, data=None):
        if not self._is_fitted:
            raise Exception('The estimator has not been fitted yet.')

        if data is None:
            wv = self._wv
        else:
            w, v = convert2array(data, self.adjustment, self.covariate)
            wv = get_wv(w, v)

        if self.combined_treat:
            return self._prepare_combined_treat(wv)
        else:
            return self._prepare_separate_treat(wv)

    def _fit_combined_treat(
        self,
        x, wv, y,
        treat,
        control,
<<<<<<< HEAD
        n_treatments,
        *args,
=======
        num_treatments,
>>>>>>> 9746a0e64a7b2ced2e52dce5d74c5c4992567ff9
        **kwargs
    ):
        treat = get_treat_control(treat, n_treatments, True)
        control = get_treat_control(control, n_treatments, False)
        self.treat = treat
        self.control = control
        self._wv = wv

        wv_treat, y_treat = get_groups(treat, x, wv, y)
        wv_control, y_control = get_groups(control, x, wv, y)

        y_treat = y_treat.squeeze()
        y_control = y_control.squeeze()

        self.xt_model.fit(wv_treat, y_treat, **kwargs)
        self.x0_model.fit(wv_control, y_control, **kwargs)

        self._is_fitted = True

        return self

    def _fit_separate_treat(
        self,
        x, wv, y,
        group_categories,
        **kwargs
    ):
        # TODO: the current implementation is astoundingly stupid
        self._fitted_dict_separa = defaultdict(list)
        waited_treat = [np.arange(len(i)) for i in group_categories]
        treat_arrays = cartesian(waited_treat)

        for treat in treat_arrays:
            model = clone(self.xt_model)
            _wv, _y = get_groups(treat, x, wv, y)
            _y = _y.squeeze()

            model.fit(_wv, _y, **kwargs)
            self._fitted_dict_separa['treatment'].append(treat)
            self._fitted_dict_separa['models'].append(model)

        self._is_fitted = True

        return self

    def _prepare_combined_treat(self, wv):
        yt = self.xt_model.predict(wv)
        y0 = self.x0_model.predict(wv)
        return yt - y0

    def _prepare_separate_treat(self, wv):
        n_treatments = len(self._fitted_dict_separa['treatment'])
        n = wv.shape[0]
        f_nji = np.full((n, self._y_d, n_treatments - 1), np.NaN)
        f_nj0 = self._fitted_dict_separa['models'][0].predict(wv)

        for i, model in enumerate(self._fitted_dict_separa['models'][1:]):
            fnji = (model.predict(wv) - f_nj0).reshape(n, self._y_d)
            f_nji[:, :, i] = fnji

        return f_nji.squeeze()


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
        **kwargs,
    ):
        assert adjustment is not None or covariate is not None, \
            'Need adjustment set or covariates to perform estimation.'

        self.outcome = outcome
        self.treatment = treatment
        self.adjustment = adjustment
        self.covariate = covariate
        self.combined_treat = combined_treatment

        y, x, w, v = convert2array(
            data, outcome, treatment, adjustment, covariate
        )
        self._y_d = y.shape[1]

        if self.categories == 'auto' or self.categories is None:
            categories = 'auto'
        else:
            categories = list(self.categories)

        self.transformer = OrdinalEncoder(categories=categories)
        self.transformer.fit(x)
        x = self.transformer.transform(x)

        group_categories = self.transformer.categories_
        n_treatments = len(group_categories)
        wv = get_wv(w, v)
        self._wv = wv

        if combined_treatment:
            return self._fit_combined_treat(
<<<<<<< HEAD
                x, wv, y, treat, control, n_treatments, *args, **kwargs
=======
                x, wv, y, treat, control, num_treatments, **kwargs
>>>>>>> 9746a0e64a7b2ced2e52dce5d74c5c4992567ff9
            )
        else:
            return self._fit_separate_treat(
                x, wv, y, group_categories, **kwargs
            )

    def _prepare4est(self, data=None, rho=0.5, *args, **kwargs):
        if not self._is_fitted:
            raise Exception('The estimator has not been fitted yet.')

        if data is None:
            wv = self._wv
        else:
            w, v = convert2array(data, self.adjustment, self.covariate)
            wv = get_wv(w, v)

        if self.combined_treat:
            effect = self._prepare_combined_treat(wv, rho)
        else:
            effect = self._prepare_separate_treat(wv, rho)

        return effect

    def estimate(
        self,
        data=None,
        rho=0.5,
        quantity='CATE',
        *args,
        **kwargs
    ):
        # TODO: add support for other types of rho
        effect = self._prepare4est(data, rho, *args, **kwargs)

        if quantity == 'CATE' or quantity == 'ATE':
            return np.mean(effect, axis=0)
        else:
            return effect

    def _fit_combined_treat(
        self,
        x, wv, y,
        treat,
        control,
<<<<<<< HEAD
        n_treatments,
        *args,
=======
        num_treatments,
>>>>>>> 9746a0e64a7b2ced2e52dce5d74c5c4992567ff9
        **kwargs
    ):
        treat = get_treat_control(treat, n_treatments, True)
        control = get_treat_control(control, n_treatments, False)
        self.treat = treat
        self.control = control

        # Step 1
        # TODO: modify the generation of group ids for efficiency
        wv_treat, y_treat = get_groups(treat, x, wv, y)
        wv_control, y_control = get_groups(control, x, wv, y)

        y_treat = y_treat.squeeze()
        y_control = y_control.squeeze()

        self.ft_model.fit(wv_treat, y_treat,  **kwargs)
        self.f0_model.fit(wv_control, y_control, **kwargs)

        # Step 2
        h_treat_target = y_treat - self.f0_model.predict(wv_treat)
        h_control_target = self.ft_model.predict(wv_control) - y_control
        self.kt_model.fit(wv_treat, h_treat_target)
        self.k0_model.fit(wv_control, h_control_target)

        # Step 3
        # This is the task of predict. See _prepare4est

        self._is_fitted = True

        return self

    def _fit_separate_treat(
        self,
        x, wv, y,
        group_categories,
        *args,
        **kwargs
    ):
        self._fitted_dict_separa = defaultdict(list)
        waited_treat = [np.arange(len(i)) for i in group_categories]
        treat_arrays = cartesian(waited_treat)

        f0_model = clone(self.f0_model)
        _wv_control, _y_control = get_groups(treat_arrays[0], x, wv, y)
        _y_control = _y_control.squeeze()
        f0_model.fit(_wv_control, _y_control)

        for treat in treat_arrays[1:]:
            ft_model = clone(self.ft_model)
            _wv, _y = get_groups(treat, x, wv, y)
            _y = _y.squeeze()

            # Step 1
            ft_model.fit(_wv, _y, *args, **kwargs)

            # Step 2
            h_treat_target = _y - f0_model.predict(_wv)
            h_control_target = ft_model.predict(_wv_control) - _y_control
            kt_model = clone(self.kt_model)
            k0_model = clone(self.k0_model)
            kt_model.fit(_wv, h_treat_target)
            k0_model.fit(_wv_control, h_control_target)

            # Step 3
            self._fitted_dict_separa['models'].append((kt_model, k0_model))
            self._fitted_dict_separa['treatment'].append(treat)

        self._is_fitted = True

        return self

    def _prepare_combined_treat(self, wv, rho):
        kt_pred = self.kt_model.predict(wv)
        k0_pred = self.k0_model.predict(wv)

        return rho * kt_pred + (1 - rho) * k0_pred

    def _prepare_separate_treat(self, wv, rho):
        model_list = self._fitted_dict_separa['models']
        n_treatments = len(self._fitted_dict_separa['treatment'])
        n = wv.shape[0]
        f_nji = np.full((n, self._y_d, n_treatments), np.NaN)

        for i, (kt_model, k0_model) in enumerate(model_list):
            pred_t = kt_model.predict(wv)
            pred_0 = k0_model.predict(wv)
            fnji = (rho * pred_t + (1 - rho) * pred_0).reshape(n, self._y_d)
            f_nji[:, :, i] = fnji

        return f_nji.squeeze()
