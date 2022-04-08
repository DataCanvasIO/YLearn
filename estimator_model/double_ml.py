from copy import deepcopy
from collections import defaultdict

import numpy as np

from sklearn import clone
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from .base_models import BaseEstLearner
from .utils import convert2array, convert4onehot, nd_kron


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

    def _prepare4est(self, data, outcome, treatment, adjustment, individual=None):
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
    # TODO: expand fij to higher orders of v.

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
        is_discrete_treatment=False,
        categories='auto',
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
            categories=categories,
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

        # TODO: the following block of code should be implemented for all fit
        # functions
        self.outcome = outcome
        self.treatment = treatment
        self.adjustment = adjustment
        self.covariate = covariate

        y, x, w, v = convert2array(
            data, outcome, treatment, adjustment, covariate
        )
        self.v = v
        self.y_d = y.shape[1]
        # random_state = self.random_state
        cfold = self.cf_fold
        n = len(data)

        if self.is_discrete_treatment:
            if self.categories == 'auto' or self.categories is None:
                categories = 'auto'
            else:
                categories = list(self.categories)

            # convert discrete treatment features to onehot vectors
            # TODO: modify this for multiple treatments
            self.transformer = OneHotEncoder(categories=categories)
            self.transformer.fit(x)
            x = self.transformer.transform(x).toarray()
        else:
            self.transformer = None

        self.x_d = x.shape[1]

        if w is None:
            wv = v
        else:
            if v is not None:
                wv = np.concatenate((w, v), axis=1)
            else:
                wv = w

        # step 1: split the data
        if cfold > 1:
            cfold = int(cfold)
            folds = [
                KFold(n_splits=cfold).split(x), KFold(n_splits=cfold).split(y)
            ]
        else:
            folds = None

        # step 2: cross fit to give the estimated y and x
        self.x_hat_dict, self.y_hat_dict = self._fit_1st_stage(
            self.x_model, self.y_model, y, x, wv, folds=folds
        )
        x_hat = self.x_hat_dict['paras'][0].reshape((n, self.x_d))
        y_hat = self.y_hat_dict['paras'][0].reshape((n, self.y_d))

        # step 3: calculate the differences
        x_diff = x - x_hat
        y_prime = y - y_hat
        x_prime = nd_kron(x_diff, v)

        # step 4: fit the regression problem
        self._fit_2nd_stage(self.yx_model, x_prime, y_prime)

        return self

    def _prepare4est(self, data=None, *args, **kwargs):
        assert self.x_hat_dict['is_fitted'][0] and \
            self.y_hat_dict['is_fitted'][0], 'x_model and y_model should be'
        'trained before estimation.'

        x_d, y_d = self.x_d, self.y_d
        v = self.v if data is None else convert2array(data, self.covariate)[0]
        n, v_d = v.shape[0], v.shape[1]

        # may need modification for multi-dim outcomes.
        # the reason we use transpose here is because coef f^i_{j, k}
        # (originally is a tensor, but here we treat it as a matrix
        # because we convert v^k x^j to a (k*j-dim) vector) has the
        # shape (y_d, x_d*v_d), the coefs for treatment in dim i are saved in
        # f[i*vd:(i+1)*vd]
        # TODO: rewrite this with einsum
        coef = self.yx_model.coef_
        # fij will be the desired treatment effect for treatment i on outcome j
        fij = np.full((n, y_d, x_d), np.NaN)

        def cal_fij(coef_vec, v):
            coef_matrix = coef_vec.reshape(-1, v_d)
            return coef_matrix.dot(v.reshape(-1, 1)).reshape(1, -1)

        # TODO: rewrite the following code with einsum for efficiency
        if y_d > 1:
            for j in range(y_d):
                for n, vn in enumerate(v):
                    fij[n, j, :] = cal_fij(coef[j], vn)
        else:
            coef_matrix = coef.reshape(-1, v_d)
            for n, vn in enumerate(v):
                fij[n, 0, :] = coef_matrix.dot(
                    vn.reshape(-1, 1)
                ).reshape(1, -1)

        return fij

    def estimate(
        self,
        data=None,
        treat=None,
        control=None,
        quantity='CATE',
    ):
        fij = self._prepare4est(data=data)
        treat = 1 if treat is None else treat
        control = 0 if control is None else control

        if self.is_discrete_treatment:
            effect = fij[:, :, treat] - fij[:, :, control]
        else:
            effect = fij * (treat - control)

        if quantity == 'CATE':
            return effect
        if quantity == 'ATE':
            return effect.mean(axis=0)

    def _cross_fit(self, model, *args, **kwargs):
        folds = kwargs.pop('folds')
        is_ymodel = kwargs.pop('is_ymodel')
        target = kwargs.pop('target')
        fitted_result = defaultdict(list)

        if not is_ymodel and self.is_discrete_treatment:
            # convert back to a vector with each dimension being a value
            # indicating the corresponding discrete value
            target_converted = convert4onehot(target)
        else:
            target_converted = target

        if folds is None:
            wv = args[0]
            model.fit(wv, target_converted)

            if not is_ymodel and self.is_discrete_treatment:
                p_hat = model.predict_proba(wv)
            else:
                p_hat = model.predict(wv)

            fitted_result['models'].append(clone(model))
            fitted_result['paras'].append(p_hat)
            idx = np.arange(start=0, stop=wv.shape[0])
            fitted_result['train_test_id'].append((idx, idx))
        else:
            fitted_result['paras'].append(np.ones_like(target) * np.nan)

            for i, (train_id, test_id) in enumerate(folds):
                model_ = clone(model)
                temp_wv = args[0][train_id]
                temp_wv_test = args[0][test_id]
                target_train = target_converted[train_id]
                model_.fit(temp_wv, target_train)

                if not is_ymodel and self.is_discrete_treatment:
                    target_predict = model_.predict_proba(temp_wv_test)
                else:
                    target_predict = model_.predict(temp_wv_test)
                    # test_shape = kwargs['target'][test_id].shape
                    # if target_predict.shape != test_shape:
                    #     target_predict.reshape(test_shape)

                fitted_result['models'].append(model_)
                fitted_result['paras'][0][test_id] = target_predict
                fitted_result['train_test_id'].append((train_id, test_id))

        fitted_result['is_fitted'] = [True]

        return fitted_result

    def _fit_1st_stage(
        self,
        x_model,
        y_model,
        y, x, wv,
        folds=None,
        **kwargs
    ):
        if self.x_d == 1:
            x = np.ravel(x)
        if self.y_d == 1:
            y = np.ravel(y)

        if folds is not None:
            x_folds, y_folds = folds
        else:
            x_folds, y_folds = None, None

        x_hat_dict = self._cross_fit(
            x_model, wv, target=x, folds=x_folds, is_ymodel=False, **kwargs
        )
        y_hat_dict = self._cross_fit(
            y_model, wv, target=y, folds=y_folds, is_ymodel=True, **kwargs
        )
        return (x_hat_dict, y_hat_dict)

    def _fit_2nd_stage(
        self,
        yx_model,
        x_prime,
        y_prime,
    ):
        yx_model.fit(x_prime, y_prime)
