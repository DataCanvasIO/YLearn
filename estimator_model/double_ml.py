from copy import deepcopy
from collections import defaultdict
from operator import mod
from statistics import mode

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn import clone

from .base_models import BaseEstLearner
from .utils import convert2array, nd_kron


class DoubleML(BaseEstLearner):
    r"""
    Double machine learning has two stages:
    In stage I, we
        1. fit a model (y_model) to predict outcome (y) from confounders (w) to
            get the predicted outcome (py);
        2. fit a model (x_model) to predict treatment (x) from confounders (w)
            to get the predicted treatement (px).
    In stage II, we
        fit a final model (yx_model) to predict y - py from x - px.

    See https://arxiv.org/pdf/1608.00060.pdf for reference.

    Attributes
    ----------

    Methods
    ----------
    _prepare(data, outcome, treatment, adjustment, individual=None)
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
    # TODO: support more final models, e.g., non-parametric models.

    def __init__(self, y_model, x_model, yx_model):
        super().__init__()
        if type(y_model) is str:
            y_model = self.ml_model_dic[y_model]
        if type(x_model) is str:
            x_model = deepcopy(self.ml_model_dic[x_model])
        if type(yx_model) is str:
            yx_model = deepcopy(self.ml_model_dic[yx_model])

        self.y_model = y_model
        self.x_model = x_model
        self.yx_model = yx_model

    def _prepare(self, data, outcome, treatment, adjustment, individual=None):
        self.y_model.fit(data[adjustment], data[outcome])
        self.x_model.fit(data[adjustment], data[treatment])

        py = self.y_model.predict(data[adjustment])
        px = self.x_model.predict(data[adjustment])

        self.yx_model.fit(data[treatment] - px, data[outcome] - py)
        # TODO: support cate, now only support ate
        result = self.yx_model.coef_
        return result

    def estimate_cate(self, data, outcome, treatment, adjustment,
                      condition_set, condition):
        raise NotImplementedError


class DML4CATE(BaseEstLearner):
    r"""Double machine learning for estimating CATE.
    # TODO: convert the einstein notations in this section to the usual ones.

    (- Skip this if you are only interested in the implementation.)
    A typical double machine learning for CATE solves the following treatment
    effect estimation (note that we use the einstein notation here):
        y^i = f^i_j(v^k) x^j + g^i(v^k, w^l) + \epsilon
        x^j = h^j(v^k, w^l) + \eta
    where f^i_j(v^k) is the CATE conditional on C=c and takes the form
        f^i_j(v^k) = F^i_{j, k} \rho^k
    with \rho^k: c \to R being v^k in the simplest case. Thus we have
        y^i = F^i_{j, k} \rho^k x^j + g^i(v^k, w^l) + \epsilon.
    The coefficients F_j^i_k can be estimated from the newly-formed data
    (\rho^k x^j, y^i) with linear regression where F^i_{j, k} are just
    coefficients of every feature in {1, 2, ..., k*j}. For a simple example, if
    both y and x only have one dimention, then the CATE for an input with
    covariate (c^1, c^2, c^3) will be F_1c^1, F_2c^2, and F_3c^3. #TODO:
    However, note that letting \rho^k simply be v^k actually implicitly assume
    that the value of v^k is small thus is a good approximation of \rho^k.


    (- Start of the implementation.)
    We implement a complicated version of the double machine learning same as
    the algorithm described in the [1]:
        1. Let k (cf_folds in our class) be an int. Form a k-fold random
        partition {..., (train_data_i, test_data_i), ...,
        (train_data_k, test_data_k)}.
        2. For each i, train y_model and x_model on train_data_i, then evaluate
        their performances in test_data_i whoes results will be saved as
        (y_hat_k, x_hat_k). All (y_hat_k, x_hat_k) will be the form
        (y_hat, x_hat).
        3. Define the differences
            y_prime = y - y_hat,
            x_prime = (x - x_hat) \otimes v.
        Then form the new dataset (y_prime, x_prime).
        4. Perform linear regression on the dataset (y_prime, x_prime) whoes
        coefficients will be saved in a vector F. The estimated CATE given V=v
        will just be
            F \dot v.
        On the other hand, the ATE can be simply estimated by taking average
        of F \dot v over the original data.

    Attributes
    ----------

    Methods
    ----------

    Reference
    ----------
    [1] V. Chernozhukov, et al. Double Machine Learning for Treatment and
        Causal Parameters. arXiv:1608.00060.
    """

    def __init__(
        self,
        x_model,
        y_model,
        yx_model=None,
        cf_fold=1,
        random_state=2022,
        is_discrete_treatment=False
    ):
        self.cf_fold = cf_fold
        self.x_model = clone(x_model)
        self.y_model = clone(y_model)

        if yx_model is None:
            self.yx_model = LinearRegression()

        self.x_hat_dict = defaultdict(list)
        self.y_hat_dict = defaultdict(list)
        self.x_hat_dict['is_fitted'].append(False)
        self.y_hat_dict['is_fitted'].append(False)

        super().__init__(
            random_state=random_state,
            is_discrete_treatment=is_discrete_treatment,
        )

    # TODO:could add a decorator in this place
    def fit(
        self,
        data,
        outcome,
        treatment,
        adjustment=None,
        covariate=None,
    ):
        """Note that we will use the following notation in this method:
                y: out

        Args:
            data (_type_): _description_
            outcome (_type_): _description_
            treatment (_type_): _description_
            adjustment (_type_, optional): _description_. Defaults to None.
            covariate (_type_, optional): _description_. Defaults to None.
        """
        assert adjustment is not None or covariate is not None, \
            'Need adjustment set or covariates to perform estimation.'

        self.outcome = outcome
        self.treatment = treatment
        self.adjustment = adjustment
        self.covariate = covariate

        y, x, w, v = convert2array(
            data, outcome, treatment, adjustment, covariate
        )
        cv = self.cf_fold
        random_state = self.random_state

        if w is None:
            wv = v
        else:
            if v is not None:
                wv = np.concatenate((w, v), axis=0)
            else:
                wv = v

        # step 1: split the data
        if cv > 1:
            cv = int(cv)
            folds = KFold(n_splits=cv, random_state=random_state).split(x, y)
        else:
            folds = None

        # step 2: cross fit to give the estimated y and x
        self.x_hat_dict, self.y_hat_dict = self._fit_1st_stage(
            self.x_model, self.y_model, y, x, wv, folds=folds
        )
        x_hat, y_hat = self.x_hat_dict['paras'], self.y_hat_dict['paras']

        # step 3: calculate the differences
        x_diff = x - x_hat
        y_prime = y - y_hat
        x_prime = nd_kron(x_diff, v)

        # step 4: fit the regression problem
        self._fit_2nd_stage(self.yx_model, x_prime, y_prime)

        return self

    def _prepare(self, data, *args, **kwargs):
        assert self.x_hat_dict['is_fitted'][0] and \
            self.y_hat_dict['is_fitted'][0], 'x_model and y_model should be'
        'trained before estimation.'

        y, x, v = convert2array(
            data, self.outcome, self.treatment, self.covariate
        )
        n, y_d, x_d, v_d = y.shape[0], y.shape[1], x.shape[1], v.shape[1]
        # may need modification for multi-dim outcomes.
        # the reason we use transpose here is because coef f^i_{j, k}
        # (originally is a tensor, but here we treat it as a matrix
        # because we convert v^k x^j to a (k*j-dim) vector) has the
        # shape (y_d, x_d*v_d), the coefs for treatment in dim i are saved in
        # f[i*vd:(i+1)*vd]
        # TODO: rewrite this with einsum
        coef = self.yx_model.coef_
        fij = np.full((n, y_d, x_d), np.NaN)
        
        def cal_fij(coef_matrix, v):
            coef_matrix.reshape(-1, v.shape[1])
            return coef_matrix.dot(v.reshape(-1, 1)).reshape(1, -1)

        for j in range(y_d):
            for n, vn in enumerate(v):
                fij[n, j, :] = cal_fij(coef[j], vn)

        return fij

    def estimate(
        self,
        data,
        treated_group=None,
        control_group=None,
        quantity='CATE'
    ):
        pass

    def _cross_fit(self, model, *args, **kwargs):
        folds = kwargs.pop['folds']
        is_ymodel = kwargs.pop['is_ymodel']
        fitted_result = defaultdict(list)

        if folds is None:
            wv = args[0]
            model.fit(wv, kwargs['target'])
            if is_ymodel and self.is_discrete_treatment:
                p_hat = model.predict_proba(wv)
            else:
                p_hat = model.predict(wv)

            fitted_result['models'].append(clone(model))
            fitted_result['paras'].append(p_hat)
            idx = np.arange(start=0, stop=wv.shape[0])
            fitted_result['train_test_id'].append((idx, idx))
        else:
            fitted_result['paras'] = np.ones_like(kwargs['target']) * np.NaN
            for i, (train_id, test_id) in enumerate(folds):
                model_ = clone(model)
                # new_args = tuple(arg[train_id] for arg in args)
                temp_wv = args[0][train_id]
                temp_wv_preidict = args[0][test_id]
                target = kwargs['target'][train_id]
                model_.fit(temp_wv, target)
                if self.is_discrete_treatment:
                    target_predict = model_.predict_proba(temp_wv_preidict)
                else:
                    target_predict = model_.predict(temp_wv_preidict)

                fitted_result['models'].append(model_)
                fitted_result['paras'][test_id] = target_predict
                fitted_result['train_test_id'].append((train_id, test_id))

        fitted_result['is_fitted'] = [True]

        return fitted_result

    def _fit_1st_stage(
        self,
        x_model,
        y_model,
        y, x, wv,
        folds=None
    ):
        x_hat_dict = self._cross_fit(
            x_model, wv, target=x, folds=folds, is_ymodel=False
        )
        y_hat_dict = self._cross_fit(
            y_model, wv, target=y, folds=folds, is_ymodel=True
        )
        return (x_hat_dict, y_hat_dict)

    def _fit_2nd_stage(
        self,
        yx_model,
        x_prime,
        y_prime,
    ):
        yx_model.fit(x_prime, y_prime)
