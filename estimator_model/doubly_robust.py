
from copy import deepcopy
from collections import defaultdict

import numpy as np

from sklearn import clone
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold


from . import propensity_score
from .base_models import BaseEstLearner
from estimator_model.utils import convert2array, get_wv, convert4onehot


class _YModelWrapper:
    def __init__(self, y_model, x_d, y_d):
        self.model = clone(y_model)
        self.x_d = x_d
        self.y_d = y_d

    def fit(self, wvx, target, **kwargs):
        self.model.fit(wvx, target, **kwargs)
        return self

    def predict(self, wvx):
        wv = wvx[:, :-self.x_d]
        n = wv.shape[0]
        y_nji = np.full((n, self.y_d, self.x_d), np.NaN)

        for i in range(self.x_d):
            x = np.zeros((n, self.x_d))
            x[:, i] == 1
            x = np.concatenate((wv, x), axis=1)
            y_nji[:, :, i] = self.model.predict(x).reshape(n, self.y_d)

        return y_nji


class DoublyRobust(BaseEstLearner):
    # TODO: consider combined treatment when dealing with multiple treatment case
    r"""
    The doubly robust estimator has 3 steps
    (see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3070495/pdf/kwq439.pdf
    for reference, see also the slides
    https://www4.stat.ncsu.edu/~davidian/double.pdf)

    The procedure is composed of 3 steps.
    1. Let k (cf_folds in our class) be an int. Form a k-fold random
    partition {..., (train_data_i, test_data_i), ...,
    (train_data_k, test_data_k)}.
    2. For each i, train y_model and x_model on train_data_i, then evaluate
    their performances in test_data_i whoes results will be saved as
    (y_hat_k, x_hat_k). All (y_hat_k, x_hat_k) will be the form
    (y_hat, x_hat). Note that for each y_hat_k, we must evaluate its values
    on all possible treatments. TODO: there might be some issuses when
    considering multi-dim y, but im not sure currently.
    3. Build the final data set (y_hat - y_hat_control, v) and train the final
    yx_model on this dataset. When estimating cate, we only use this final
    model to predict v.
    """

    def __init__(
        self,
        x_model,
        y_model,
        yx_model,
        cf_fold=1,
        random_state=2022,
        categories='auto',
    ):
        self.cf_fold = cf_fold
        self.x_model = clone(x_model)
        self.y_model = clone(y_model)
        self.yx_model = clone(yx_model)

        self.x_hat_dict = defaultdict(list)
        self.y_hat_dict = defaultdict(list)
        self.x_hat_dict['is_fitted'].append(False)
        self.y_hat_dict['is_fitted'].append(False)

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
    ):
        assert adjustment is not None or covariate is not None, \
            'Need adjustment or covariate to perform estimation.'
        super().fit(data, outcome, treatment,
                    adjustment=adjustment,
                    covariate=covariate,
                    )
        n = len(data)
        y, x, w, v = convert2array(
            data, outcome, treatment, adjustment, covariate
        )

        self._v = v
        cfold = self.cf_fold

        if self.categories == 'auto' or self.categories is None:
            categories = 'auto'
        else:
            categories = list(self.categories)

        self.transformer = OneHotEncoder(categories=categories)
        self.transformer.fit(x)
        x = self.transformer.transform(x).toarray()
        self._x_d = x.shape[1]
        self._y_d = y.shape[1]
        wv = get_wv(w, v)
        y_model = self._gen_y_model(self.y_model)

        # step 1
        if cfold > 1:
            cfold = int(cfold)
            folds = [
                KFold(n_splits=cfold).split(x), KFold(n_splits=cfold).split(y)
            ]
        else:
            folds = None

        # step 2: cross fit to give the estimated y and x
        self.x_hat_dict, self.y_hat_dict = self._fit_1st_stage(
            self.x_model, y_model, y, x, wv, folds=folds
        )
        x_hat = self.x_hat_dict['paras'][0].reshape((x.shape))
        y_hat = self.y_hat_dict['paras'][0]  # y_hat has shape (n, j, i)

        # step 3
        # calculate the estimated y
        y_prime = np.full((n, self._y_d, self._x_d), np.NaN)

        for i in range(self._x_d):
            y_nji = y_hat[:, :, i]
            x_hat_i = x_hat[:, i].reshape(-1, 1)
            # print(np.mean(x[:, i] - x_hat[:, i], axis=0))
            y_nji += ((y - y_nji) * (x[:, i] ==
                      1).reshape(-1, 1)) / (x_hat_i + 1e-5)
            y_prime[:, :, i] = y_nji

        self._final_result = self._fit_2nd_stage(self.yx_model, v, y_prime)
        return self

    def estimate(
        self,
        data=None,
        control=None,
        quantity='CATE',
    ):
        y_pred_nji = self._prepare4est(data=data)
        control = 0 if control is None else control
        y_pred_control = y_pred_nji[:, :, control]

        effects = []
        for i in range(self._x_d):
            effects.append(y_pred_nji[:, :, i] - y_pred_control)

        if quantity == 'CATE':
            assert self._v is not None, 'Need covariates to estimate CATE.'
            return [effect.mean(axis=0) for effect in effects]
        elif quantity == 'ATE':
            return [effect.mean(axis=0) for effect in effects]
        else:
            return effects

    def _prepare4est(self, data=None):
        if not all((self.x_hat_dict['is_fitted'][0],
                   self.y_hat_dict['is_fitted'][0],
                   self._final_result['is_fitted'][0])):
            raise Exception('The model has not been fully fitted yet.')

        v = self._v if data is None else convert2array(data, self.covariate)[0]

        if v is None:
            y_pred_nji = self._final_result['effect']
        else:
            n = v.shape[0]
            y_pred_nji = np.full((n, self._y_d, self._x_d), np.NaN)

            for i in range(self._x_d):
                model = self._final_result['models'][i]
                y_pred_nji[:, :, i] = model.predict(v).reshape(n, self._y_d)

        return y_pred_nji

    def _fit_1st_stage(
        self,
        x_model,
        y_model,
        y, x, wv,
        folds=None,
        **kwargs
    ):
        y = y.squeeze()

        if folds is not None:
            x_folds, y_folds = folds
        else:
            x_folds, y_folds = None, None

        wvx = np.concatenate((wv, x), axis=1)
        # convert back to a vector with each dimension being a value
        # indicating the corresponding discrete value
        label = convert4onehot(x)

        x_hat_dict = self._cross_fit(
            x_model, wv, target=label, folds=x_folds, is_ymodel=False, **kwargs
        )
        y_hat_dict = self._cross_fit(
            y_model, wvx, target=y, folds=y_folds, is_ymodel=True, **kwargs
        )

        return x_hat_dict, y_hat_dict

    def _fit_2nd_stage(
        self,
        yx_model,
        v,
        y_prime,
    ):
        final_result = defaultdict(list)
        final_result['is_fitted'].append(False)

        if v is None:
            final_result['effect'] = y_prime
        else:
            for i in range(self._x_d):
                model = clone(yx_model)
                model.fit(v, y_prime[:, :, i].squeeze())
                final_result['models'].append(model)

        final_result['is_fitted'][0] = True
        return final_result

    def _cross_fit(self, model, *args, **kwargs):
        folds = kwargs.pop('folds', None)
        is_ymodel = kwargs.pop('is_ymodel', False)
        target = kwargs.pop('target')
        fitted_result = defaultdict(list)

        if folds is None:
            wv = args[0]
            model.fit(wv, target, **kwargs)

            if not is_ymodel:
                p_hat = model.predict_proba(wv)
            else:
                p_hat = model.predict(wv)

            fitted_result['models'].append(deepcopy(model))
            fitted_result['paras'].append(p_hat)
            idx = np.arange(start=0, stop=wv.shape[0])
            fitted_result['train_test_id'].append((idx, idx))
        else:
            for i, (train_id, test_id) in enumerate(folds):
                model_ = deepcopy(model)
                temp_wv = args[0][train_id]
                temp_wv_test = args[0][test_id]
                target_train = target[train_id]
                model_.fit(temp_wv, target_train, **kwargs)
                n, y_d, x_d = target.shape[0], self._y_d, self._x_d

                if not is_ymodel:
                    target_predict = model_.predict_proba(temp_wv_test)
                    if i == 0:
                        target_required = np.full((n, x_d), np.NaN)
                        fitted_result['paras'].append(target_required)
                else:
                    target_predict = model_.predict(temp_wv_test)
                    if i == 0:
                        target_required = np.full((n, y_d, x_d), np.NaN)
                        fitted_result['paras'].append(target_required)

                fitted_result['models'].append(model_)
                fitted_result['paras'][0][test_id] = target_predict
                fitted_result['train_test_id'].append((train_id, test_id))

        fitted_result['is_fitted'] = [True]

        return fitted_result

    def _gen_x_model(self):
        # if use x_model without methods like fit, predict_proba...
        # may have to define new wrappers
        # TODO
        pass

    def _gen_y_model(self, model):
        y_model = clone(model)
        return _YModelWrapper(y_model=y_model, x_d=self._x_d, y_d=self._y_d)


class _DoublyRobustOld(BaseEstLearner):
    r"""
    The doubly robust estimator has 3 steps
    (see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3070495/pdf/kwq439.pdf
    for reference, see also the slides
    https://www4.stat.ncsu.edu/~davidian/double.pdf):
    1. Split the data into the treatment group (xt_data) and control group
        (x0_data),then fit two models in these groups to predict the outcome y:
       yt_i = xt_model.predict(w_i), y0_i = x0_model.predict(w_i)
    2. Estimate the propensity score
       ps(w_i) = ps_model.predict(w_i)
    3. Calculate the final result (expected result, note that ps_model should
        be a multi-classification for discrete treatment):
        1/n \sum_i^n [
            (\frac{I(x_i=xt)y_i}{ps_{x_i=xt}(w_i)}
            - yt_i\frac{I(x_i=xt)-ps_{x_i=xt}(w_i)}{ps_{x_i=xt}(w_i)})
            (\frac{I(x_i=x0)y_i}{ps_{x_i=x0}(w_i)}
            - y0_i\frac{I(x_i=x0)-ps_{x_i=x0}(w_i)}{ps_{x_i=x0}(w_i)})
        ]

    Attributes
    ----------
    ml_model_dic : dict
        A dictionary of default machine learning sklearn models currently
        including
            'LR': LinearRegression
            'LogisticR': LogisticRegression.
    ps_model : PropensityScore
    xt_model : MLModel, optional
        The machine learning model trained in the treated group.
    x0_model : MLModel, optional
        The machine learning model trained in the control group.

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

    def __init__(self, ps_model='LogisticR', est_model='LR'):
        """
        Parameters
        ----------
        ps_model : PropensityScore or str
        est_model : str, optional
            All valid est_model should implement the fit and predict methods
            which can be done by wrapping the corresponding machine learning
            models with the class MLModel.
        """
        super().__init__()

        if type(ps_model) is str:
            ps_model = self.ml_model_dic[ps_model]
        if type(est_model) is str:
            est_model = self.ml_model_dic[est_model]

        # TODO: should we train different regression models for different
        # treatment groups or simply train one model once to model the
        # relation between treatment and adjustment?
        self.ps_model = propensity_score.PropensityScore(ml_model=ps_model)
        self.xt_model = est_model
        self.x0_model = deepcopy(est_model)

    def _prepare4est(self, data, outcome, treatment, adjustment,
                     individual=None, treatment_value=None):
        # TODO: categorical treatment. Currently I hope to convert categorical
        # treatments to integers such that treatment={1, 2,...,n} where n is
        # the number of different treatments. Is there any better idea?

        # The default treatment group data are those whose data[treatment] == 1
        num_treatment = data[treatment].value_counts().shape[0]
        if treatment_value is None or num_treatment == 2:
            treatment_value = 1

        # step 1, fit the treatment group model and the control group model
        xt_data = data.loc[data[treatment] == treatment_value]
        x0_data = data.loc[data[treatment] == 0]
        self.xt_model.fit(xt_data[adjustment], xt_data[outcome])
        self.x0_model.fit(x0_data[adjustment], x0_data[outcome])

        # step 2, fit the propensity score model
        self.ps_model.fit(data, treatment, adjustment)

        # step 3, calculate the final result
        if individual:
            data_ = individual
        else:
            data_ = data
        yt = self.xt_model.predict(data_[adjustment])
        y0 = self.x0_model.predict(data_[adjustment])
        x, y = data_[treatment], data_[outcome]
        # eps = 1e-7

        x0_index = (x == 0).astype(int)
        x0_prob = self.ps_model.predict_proba(data_, adjustment, 0)
        xt_index = (x == treatment_value).astype(int)
        xt_prob = self.ps_model.predict_proba(
            data_, adjustment, treatment_value
        )
        result = (
            (y * xt_index / xt_prob + (xt_prob - xt_index) * yt / xt_prob)
            - (y * x0_index / x0_prob + (x0_prob - x0_index) * y0 / x0_prob)
        )
        return result
