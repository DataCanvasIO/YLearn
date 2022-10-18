from copy import deepcopy
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn import clone
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from .base_models import BaseEstModel
from .utils import (
    check_classes,
    convert2array,
    convert4onehot,
    nd_kron,
    get_wv,
    cartesian,
    get_tr_ctrl,
)
from ylearn.utils import logging

logger = logging.get_logger(__name__)

#
# class DoubleML(BaseEstModel):
#     r"""
#     Double machine learning has two stages:
#     In stage I, we
#         1. fit a model (y_model) to predict outcome (y) from confounders (w) to
#             get the predicted outcome (py);
#         2. fit a model (x_model) to predict treatment (x) from confounders (w)
#             to get the predicted treatement (px).
#     In stage II, we
#         fit a final model (yx_model) to predict y - py from x - px.
#
#     See https://arxiv.org/pdf/1608.00060.pdf for reference.
#
#     Attributes
#     ----------
#
#     Methods
#     ----------
#     _prepare4est(data, outcome, treatment, adjustment, individual=None)
#         Prepare (fit the model) for estimating various quantities including
#         ATE, CATE, ITE, and CITE.
#     estimate(data, outcome, treatment, adjustment, quantity='ATE',
#                  condition_set=None, condition=None, individual=None)
#         Integrate estimations for various quantities into a single method.
#     estimate_ate(self, data, outcome, treatment, adjustment)
#     estimate_cate(self, data, outcome, treatment, adjustment,
#                       condition_set, condition)
#     estimate_ite(self, data, outcome, treatment, adjustment, individual)
#     estimate_cite(self, data, outcome, treatment, adjustment,
#                       condition_set, condition, individual)
#     """
#     # TODO: support more final models, e.g., non-parametric models.
#
#     def __init__(self, y_model, x_model, yx_model):
#         super().__init__()
#         if type(y_model) is str:
#             y_model = self.ml_model_dic[y_model]
#         if type(x_model) is str:
#             x_model = deepcopy(self.ml_model_dic[x_model])
#         if type(yx_model) is str:
#             yx_model = deepcopy(self.ml_model_dic[yx_model])
#
#         self.y_model = y_model
#         self.x_model = x_model
#         self.yx_model = yx_model
#
#     def _prepare4est(self, data, outcome, treatment, adjustment, individual=None):
#         self.y_model.fit(data[adjustment], data[outcome])
#         self.x_model.fit(data[adjustment], data[treatment])
#
#         py = self.y_model.predict(data[adjustment])
#         px = self.x_model.predict(data[adjustment])
#
#         self.yx_model.fit(data[treatment] - px, data[outcome] - py)
#         # TODO: support cate, now only support ate
#         result = self.yx_model.coef_
#         return result
#
#     def estimate_cate(self, data, outcome, treatment, adjustment,
#                       condition_set, condition):
#         raise NotImplementedError


def _set_random_state(model, random_state):
    if (
        hasattr(model, "set_params")
        and hasattr(model, "random_state")
        and (model.random_state is None)
    ):
        model.set_params(random_state=random_state)


class DoubleML(BaseEstModel):
    r"""Double machine learning for estimating CATE.
    # TODO: convert the einstein notations in this section to the usual ones.
    # TODO: expand fij to higher orders of v.

    (- Skip this if you are only interested in the implementation.)
    A typical double machine learning for CATE solves the following treatment
    effect estimation (note that we use the einstein notation here):
        y^i = f^i_j(v^k) x^j + g^i(v^k, w^l) + \epsilon
        x^j = h^j(v^k, w^l) + \eta
    where f^i_j(v^k) is the CATE conditional on V=v and takes the form
        f^i_j(v^k) = F^i_{j, k} \rho^k
    with \rho^k: v \to R being v^k in the simplest case. Thus we have
        y^i = F^i_{j, k} \rho^k x^j + g^i(v^k, w^l) + \epsilon.
    The coefficients F_j^i_k can be estimated from the newly-formed data
    (\rho^k x^j, y^i) with linear regression where F^i_{j, k} are just
    coefficients of every feature in {1, 2, ..., k*j}. For a simple example, if
    both y and x only have one dimention, then the CATE for an input with
    covariate (v^1, v^2, v^3) will be F_1v^1, F_2v^2, and F_3v^3. #TODO:
    However, note that letting \rho^k simply be v^k actually implicitly assumes
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

    treat : float or ndarray
        In the case of single discrete treatment, treat should be an int or
        str in one of all possible treatment values which indicates the
        value of the intended treatment;
        in the case of multiple discrete treatment, treat should be a list
        or an ndarray where treat[i] indicates the value of the i-th intended
        treatment;
        in the case of continuous treatment, treat should be a float or a
        ndarray, by default None

    _v : np.array
        Covariate variables in the training set.

    _y_d : int
        Dimension of the outcome.

    _x_d : int
        Dimension of the treatment.

    ord_transformer : OrdinalEncoder
        Ordinal transformer of the discrete treament.

    oh_transformer : OneHotEncoder
        One hot encoder of the discrete treatment. Note that the total transformer
        is combined by the ord_transformer and oh_transformer. See comp_transformer
        for detail.

    label_dict : dict

    x_hat_dict : defaultdict(list)
        Cached values when fitting the treatment model.

    y_hat_dict : defaultdict(list)
        Cached values when fitting the outcome model.

    Methods
    ----------
    fit(data, outcome, treatment, adjustment, covariate)
        Fit the DoubleML estimator model.

    estimate(data, treat, control, quantity)
        Estimate the causal effect.

    comp_transformer(x, categories='auto')
        Transform the discrete treatment into one-hot vectors.

    _cross_fit(model)
        Fit x_model and y_model in a cross fitting manner.

    _fit_first_stage(x_model, y_model, y, x, wv, folds)
        Fit the first stage of the double machine learning.

    _fit_second_stage(yx_model, y_prime, x_prime)
        Fit the second stage of the DML.

    _prepare4est(data)

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
        adjustment_transformer=None,
        covariate_transformer=None,
        random_state=2022,
        is_discrete_treatment=False,
        categories="auto",
        is_discrete_outcome=False,
        proba_output=False,
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
        if proba_output:
            assert (
                is_discrete_outcome
            ), f"proba_output requires is_discrete_outcome to be True but was given {is_discrete_outcome}"
            assert hasattr(
                y_model, "predict_proba"
            ), f"The predict_proba method of {y_model} is required to use proba_output. But None was given."
        self.proba_output = proba_output
        self.y_pred_func = "predict_proba" if proba_output else "predict"

        if is_discrete_treatment:
            assert hasattr(
                x_model, "predict_proba"
            ), f"The predict_proba method of {x_model} is required when is_discrete_treatment is True."
            self.x_pred_func = "predict_proba"
        else:
            self.x_pred_func = "predict"

        self.cf_fold = cf_fold
        self.x_model = clone(x_model)
        self.y_model = clone(y_model)

        _set_random_state(self.x_model, random_state)
        _set_random_state(self.y_model, random_state)

        if yx_model is None:
            self.yx_model = LinearRegression()
        else:
            self.yx_model = yx_model

        self.adjustment_transformer = adjustment_transformer
        self.covariate_transformer = covariate_transformer

        self.x_hat_dict = defaultdict(list)
        self.y_hat_dict = defaultdict(list)

        self.x_hat_dict["is_fitted"].append(False)
        self.y_hat_dict["is_fitted"].append(False)

        super().__init__(
            random_state=random_state,
            is_discrete_treatment=is_discrete_treatment,
            is_discrete_outcome=is_discrete_outcome,
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
        **kwargs,
    ):
        """Fit the DoubleML estimator model.

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

        Returns
        -------
        instance of DoubleML
            The fitted estimator model
        """
        # must have adjustment to evaluate ATE,
        # must also have covaraite to evalueate CATE
        assert (
            adjustment is not None or covariate is not None
        ), "Need adjustment set or covariates to perform estimation."

        super().fit(
            data,
            outcome,
            treatment,
            adjustment=adjustment,
            covariate=covariate,
            **kwargs,
        )

        y, x, w, v = convert2array(data, outcome, treatment, adjustment, covariate)
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
            # note that when there are multiple treatments, we should convert
            # the problem into a single discrete treatment problem
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
        self.x_hat_dict, self.y_hat_dict = self._fit_1st_stage(
            self.x_model, self.y_model, y, x, wv, folds=folds
        )
        x_hat = self.x_hat_dict["paras"][0].reshape((x.shape))
        if self.proba_output:
            assert y.shape[1] == 1, "Currently only support one discrete outcome."
            self._outcome_oh = OneHotEncoder(
                categories=[self.y_hat_dict["models"][0].classes_]
            )  # TODO: note that when the cfold is too large, the classes_ may not include all valid classes
            y = self._outcome_oh.fit_transform(y)
        y_hat = self.y_hat_dict["paras"][0].reshape(y.shape)

        # step 3: calculate the differences
        y_prime = y - y_hat
        x_prime = self._cal_x_prime(x, x_hat, v)

        # step 4: fit the regression problem
        self._fit_2nd_stage(self.yx_model, x_prime, y_prime, v, **kwargs)

        self._is_fitted = True

        return self

    def estimate(
        self,
        data=None,
        treat=None,
        control=None,
        quantity=None,
        target_outcome=None,
    ):
        """Estimate the causal effect.

        Parameters
        ----------
        data : pd.DataFrame, optional
            The test data for the estimator to evaluate the causal effect, note
            that the estimator directly evaluate all quantities in the training
            data if data is None, by default None

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

        quantity : str, optional
            The possible values of quantity include:
                'CATE' : the estimator will evaluate the CATE;
                'ATE' : the estimator will evaluate the ATE;
                None : the estimator will evaluate the ITE or CITE, by default None

        Returns
        -------
        ndarray
            The estimated causal effect with the type of the quantity.
        """
        fij = self._prepare4est(data=data)
        if hasattr(self, "treat") and treat is None:
            treat = self.treat
        if hasattr(self, "control") and control is None:
            control = self.control

        dis_tr = self.is_discrete_treatment

        if isinstance(treat, pd.Series):
            treat = treat.values

        if isinstance(control, pd.Series):
            control = control.values

        if not isinstance(treat, np.ndarray):
            treat = get_tr_ctrl(
                treat,
                self.comp_transormer,
                treat=True,
                one_hot=False,
                discrete_treat=dis_tr,
            )

        if not isinstance(control, np.ndarray):
            control = get_tr_ctrl(
                control,
                self.comp_transormer,
                treat=False,
                one_hot=False,
                discrete_treat=dis_tr,
            )

        self.treat = treat

        if self.is_discrete_treatment:
            effect = fij[:, :, treat] - fij[:, :, control]
        else:
            if isinstance(treat, np.ndarray):
                treat = treat.reshape(-1, self._x_d)
            if isinstance(control, np.ndarray):
                control = control.reshape(-1, self._x_d)
                effect = np.einsum("nji, ni->nji", fij, treat - control)
            else:
                effect = fij * (treat - control)

        if target_outcome is not None:
            assert (
                self.proba_output
            ), f"target_outcome can only be specificed when proba_output is True."
            target_outcome = check_classes(target_outcome, self.outcome_classes_)
            if effect.ndim == 3:
                effect = effect[:, target_outcome, :].reshape(effect.shape[0], 1, -1)
            else:
                effect = effect[:, target_outcome]

        if quantity == "CATE":
            assert self.covariate is not None
            return effect.mean(axis=0)
        elif quantity == "ATE":
            return effect.mean(axis=0)
        else:
            return effect

    def comp_transormer(self, x, categories="auto"):
        """Transform the discrete treatment into one-hot vectors.

        Parameters
        ----------
        x : ndarray, shape (n, x_d)
            An array containing the information of the treatment variables

        categories : str or list, optional
            by default 'auto'

        Returns
        -------
        ndarray
            The transformed one-hot vectors
        """
        if x.shape[1] > 1:
            if not self._is_fitted:
                self.ord_transformer = OrdinalEncoder(categories=categories)
                self.ord_transformer.fit(x)

                labels = [np.arange(len(c)) for c in self.ord_transformer.categories_]
                labels = cartesian(labels)
                categories = [np.arange(len(labels))]

                self.label_dict = {tuple(k): i for i, k in enumerate(labels)}

            x_transformed = self.ord_transformer.transform(x).astype(int)
            x = np.full((x.shape[0], 1), np.NaN)

            # time consuming!
            for i, x_i in enumerate(x_transformed):
                x[i] = self.label_dict[tuple(x_i)]

        if not self._is_fitted:
            self.oh_transformer = OneHotEncoder(categories=categories)
            self.oh_transformer.fit(x)

        x = self.oh_transformer.transform(x).toarray()

        return x

    def effect_nji(self, data=None, control=None):
        y_nji = self._prepare4est(data=data)
        n, x_d = y_nji.shape[0], y_nji.shape[2]

        if self.is_discrete_treatment:
            if hasattr(self, "control") and control is None:
                control = self.control
            control = get_tr_ctrl(
                control,
                self.comp_transormer,
                treat=False,
                one_hot=False,
                discrete_treat=self.is_discrete_treatment,
            )
            temp_y = y_nji[:, :, control].reshape(n, -1, 1)
            temp_y = np.repeat(temp_y, x_d, axis=2)
            y_nji = y_nji - temp_y

        return y_nji

    def _prepare4est(self, data=None, *args, **kwargs):
        """Prepare for the estimation of causal quantities.

        Parameters
        ----------
        data : pd.DataFrame, optional
            The test data for evaluating the causal quantities, note that
            the estimator will perform the estimation in the training data if
            data is None, by default None

        Returns
        -------
        ndarray, shape (n, y_d, x_d)
            The element in the slot of (i, j, k) indicating the causal effect of
            treatment k on the j-th dimension of the outcome for the i-th example
        """
        assert all(
            (
                self.x_hat_dict["is_fitted"][0],
                self.y_hat_dict["is_fitted"][0],
                self._is_fitted,
            )
        ), "x_model and y_model should be trained before estimation."
        x_d = self._x_d
        if self.proba_output:
            y_d = self.outcome_classes_.__len__()
        else:
            y_d = self._y_d
        v = self._v if data is None else convert2array(data, self.covariate)[0]

        if self.covariate_transformer is not None and v is not None:
            v = self.covariate_transformer.transform(v)
        v = np.hstack([np.ones((v.shape[0], 1)), v])
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
                fij[n, 0, :] = coef_matrix.dot(vn.reshape(-1, 1)).reshape(1, -1)

        return fij

    @property
    def outcome_classes_(self):
        if self.is_discrete_outcome:
            assert self._is_fitted, "The model has not been fitted yet."
            return self.y_model.classes_
        else:
            return None

    def _cross_fit(self, model, *args, **kwargs):
        folds = kwargs.pop("folds")
        is_ymodel = kwargs.pop("is_ymodel")
        target = kwargs.pop("target")
        fitted_result = defaultdict(list)

        if not is_ymodel and self.is_discrete_treatment:
            # convert back to a vector with each dimension being a value
            # indicating the corresponding discrete value
            target_converted = convert4onehot(target)
            pred_func = self.x_pred_func
        else:
            target_converted = target
            pred_func = self.y_pred_func

        if folds is None:
            wv = args[0]
            model.fit(wv, target_converted, **kwargs)

            p_hat = model.__getattribute__(pred_func)(wv)

            fitted_result["models"].append(model)
            fitted_result["paras"].append(p_hat)
            idx = np.arange(start=0, stop=wv.shape[0])
            fitted_result["train_test_id"].append((idx, idx))
        else:
            fitted_result["paras"].append(np.ones_like(target) * np.nan)

            for i, (train_id, test_id) in enumerate(folds):
                model_ = clone(model)
                temp_wv = args[0][train_id]
                temp_wv_test = args[0][test_id]
                target_train = target_converted[train_id]
                model_.fit(temp_wv, target_train, **kwargs)
                target_predict = model_.__getattribute__(pred_func)(temp_wv_test)

                fitted_result["models"].append(model_)
                fitted_result["paras"][0][test_id] = target_predict
                fitted_result["train_test_id"].append((train_id, test_id))

        fitted_result["is_fitted"] = [True]

        return fitted_result

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

    def _fit_2nd_stage(
        self,
        yx_model,
        x_prime,
        y_prime,
        v,
        **kwargs,
    ):
        """Fit the models in the second stage.

        Parameters
        ----------
        yx_model : estimator
            _description_

        x_prime : ndarray, shape (n, x_d)
            The residuls of the treatment vector x

        y_prime : ndarray, shape (n, y_d)
            The residuls of the outcome vector y
        """
        logger.info(f"_fit_2nd_stage: fitting yx_model {type(self.yx_model).__name__}")
        yx_model.fit(x_prime, y_prime)

    def _cal_x_prime(self, x, x_hat, v):
        x_diff = x - x_hat
        v = np.hstack([np.ones((v.shape[0], 1)), v])
        return nd_kron(x_diff, v)

    # def __repr__(self) -> str:
    #     return f'Double Machine Learning Estimator'
