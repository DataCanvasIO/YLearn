
from copy import deepcopy
from collections import defaultdict

import numpy as np

from sklearn import clone
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import KFold


from . import propensity_score
from .base_models import BaseEstModel
from .utils import (convert2array, get_wv, convert4onehot,
                    cartesian, get_tr_ctrl)


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


class DoublyRobust(BaseEstModel):
    # TODO: consider combined treatment when dealing with multiple treatment case
    r"""
    The doubly robust estimator has 3 steps
    (see [1] for reference, see also the slides [2])

    The procedure is composed of 3 steps.
    1. Let k (cf_folds in our class) be an int. Form a k-fold random
    partition {..., (train_data_i, test_data_i), ...,
    (train_data_k, test_data_k)}.

    2. For each i, train y_model and x_model on train_data_i, then evaluate
    their performances in test_data_i whoes results will be saved as
    (y_hat_k, x_hat_k). Note that x_model should be a machine learning model
    with a predict_proba method. All (y_hat_k, x_hat_k) will be combined to
    give the new dataset (y_hat, x_hat). Note that for each y_hat_k, we must
    evaluate its values on all possible treatments, i.e., y_hat should have 
    the shape (n, y_d, x_d) where n is the number of examples, y_d is the
    dimension of y, and x_d is the dimension of x (the number of possible
    treatment values). Note that if there are multiple treatments, we can
    either use the combined_treatment technique to covert the multiple
    discrete classification tasks into a single discrete classification task
    or simply train multiple models for each classification task. For an
    example, if there are two different binary treatments:
        treatment_1: x_1 | x_1 \in {'sleep', 'run'},
        treatment_2: x_2 | x_2 \in {'study', 'work'},
    then we can convert to these two binary classification tasks into a single
    classification with 4 different classes:
        treatment: x | x \in {0, 1, 2, 3},
    where, for example, 1 stands for ('sleep' and 'stuy').
    TODO: there might be some issuses when considering multi-dim y, but im not
    sure currently.

    3. Build the final dataset (v, y_prime_treat - y_prime_control) where y_prime
    is difined as
        y_prime_treat = y_hat_x + \frac{(y - y_hat_x) * I(X=x)}{x_hat_x}
    and train the final yx_model on this dataset to predict the causal effect, 
        y_hat_treat - y_hat_control,
    on v. Then we can directly estimate the CATE by passing the covariate v to
    the model. Note that we train several independent models for each value of
    treatment, thus if there are i different values for treatment (use the
    combined_treatment technique for multiple treatments) we will have i - 1
    different models.

    Attributes
    ----------
    _is_fitted : bool
        True if the model is fitted ortherwise False.

    x_model : estimator
        Any valid x_model should implement the fit and predict_proba methods

    y_model : estimator
        Any valid y_model should implement the fit and predict methods

    yx_model : estimatro
        Any valid yx_model should implement the fit and predict methods

    cf_fold : int, optional
        The nubmer of folds for performing cross fit, by default 1

    random_state : int, optional
        Random seed, by default 2022

    categories : str, optional
        by default 'auto'

    x_hat_dict : dict
        Cached values when fitting the x_model.

    y_hat_dict : dict
        Cached values when fitting the y_model.

    ord_transformer : OrdinalEncoder
        Ordinal transformer of the discrete treament.

    oh_transformer : OneHotEncoder
        One hot encoder of the discrete treatment. Note that the total transformer
        is combined by the ord_transformer and oh_transformer. See comp_transformer
        for detail.

    label_dict : dict        

    Methods
    ----------
    fit(data, outcome, treatment, adjustment, covariate)
        Fit the DoublyRobust estimator model.

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

    _gen_x_model()

    _gen_y_model

    Reference
    ----------
    [1] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3070495/pdf/kwq439.pdf
    [2] https://www4.stat.ncsu.edu/~davidian/double.pdf 
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
        """
        Parameters
        ----------
        x_model : estimator
            The machine learning model which is trained to modeling the treatment. 
            Any valid x_model should implement the :py:func:`fit()` and :py:func:`predict_proba()` methods

        y_model : estimator
            The machine learning model which is trained to modeling the outcome with covariates (possibly adjustment) and the  treatment. 
            Any valid y_model should implement the fit and predict methods

        yx_model : estimatro
            The machine learning model which is trained in the final stage of doubly robust method to modeling the causal effects with covariates (possibly adjustment). 
            Any valid yx_model should implement the fit and predict methods

        cf_fold : int, optional
            The nubmer of folds for performing cross fit in th first stage, by default 1

        random_state : int, optional
            Random seed, by default 2022

        categories : str, optional
            by default 'auto'
        """
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
            is_discrete_treatment=True,
        )

    def fit(
        self,
        data,
        outcome,
        treatment,
        treat=None,
        control=None,
        adjustment=None,
        covariate=None,
    ):
        """Fit the DoublyRobust estimator model. Note that the trainig of a doubly robust model has three stages, where we implement them in 
        :py:func:`_fit_1st_stage` and :py:func:`_fit_2nd_stage`.

        Parameters
        ----------
        data : pandas.DataFrame
            The dataset used for training the model

        outcome : str or list of str, optional
            Names of the outcome variables

        treatment : str or list of str
            Names of the treatment variables

        treat : float or ndarray, optional
            In the case of single discrete treatment, treat should be an int or
            str in one of all possible treatment values which indicates the
            value of the intended treatment;
            in the case of multiple discrete treatment, treat should be a list
            or an ndarray where treat[i] indicates the value of the i-th intended
            treatment. For example, when there are multiple
            discrete treatments, array(['run', 'read']) means the treat value of
            the first treatment is taken as 'run' and that of the second treatment
            is taken as 'read'.

        control : float or ndarray, optional
            This is similar to the cases of treat, by default None

        adjustment : str or list of str, optional
            Names of the adjustment variables, by default None

        covariate : str or list of str, optional
            Names of the covariate variables, by default None

        Returns
        -------
        instance of DoublyRobust
            The fitted estimator model.
        """
        assert adjustment is not None or covariate is not None, \
            'Need adjustment or covariate to perform estimation.'
        super().fit(
            data, outcome, treatment,
            adjustment=adjustment,
            covariate=covariate,
        )

        # get numpy data
        n = len(data)
        y, x, w, v = convert2array(
            data, outcome, treatment, adjustment, covariate
        )

        self._v = v
        # get the number of cross fit folds
        cfold = self.cf_fold

        if self.categories == 'auto' or self.categories is None:
            categories = 'auto'
        else:
            categories = list(self.categories)

        # transform x into one_hot vectors
        x = self.comp_transormer(x, categories=categories)
        self._x_d = x.shape[1]
        self._y_d = y.shape[1]

        wv = get_wv(w, v)
        # generate y model
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
        y_prime = np.full((n, y.shape[1], x.shape[1]), np.NaN)

        for i in range(self._x_d):
            y_nji = y_hat[:, :, i]
            x_hat_i = x_hat[:, i].reshape(-1, 1)
            # print(np.mean(x[:, i] - x_hat[:, i], axis=0))
            y_nji += ((y - y_nji) * (x[:, i] ==
                      1).reshape(-1, 1)) / (x_hat_i + 1e-5)
            y_prime[:, :, i] = y_nji

        self._final_result = self._fit_2nd_stage(
            self.yx_model, v, y_prime, control, treat
        )

        self._is_fitted = True

        return self

    def estimate(
        self,
        data=None,
        quantity=None,
        treat=None,
        all_tr_effects=False,
    ):
        """Estimate the causal effect.

        Parameters
        ----------
        data : pd.DataFrame, optional
            The test data for the estimator to evaluate the causal effect, note
            that the estimator directly evaluate all quantities in the training
            data if data is None, by default None

        quantity : str, optional
            The possible values of quantity include:
                'CATE' : the estimator will evaluate the CATE;
                'ATE' : the estimator will evaluate the ATE;
                None : the estimator will evaluate the ITE or CITE, by default None

        all_tr_effects : bool
            If True, return all treatment effects with different treatments, otherwise
            only return the treatment effect of the treatment with the value of 
            treat if treat is provided. If treat is not provided, then the value of
            treatment is taken as the value of that when fitting
            the estimator model.

        treat : float or ndarray, optional
            In the case of single discrete treatment, treat should be an int or
            str in one of all possible treatment values which indicates the
            value of the intended treatment;
            in the case of multiple discrete treatment, treat should be a list
            or an ndarray where treat[i] indicates the value of the i-th intended
            treatment. For example, when there are multiple
            discrete treatments, array(['run', 'read']) means the treat value of
            the first treatment is taken as 'run' and that of the second treatment
            is taken as 'read'.

        Returns
        -------
        ndarray
            The estimated causal effect with the type of the quantity.
        """
        # shape (n, y_d, x_d)
        y_pred_nji = self._prepare4est(
            data=data, all_tr=all_tr_effects, treat=treat
        )

        if quantity == 'CATE':
            assert self._v is not None, 'Need covariates to estimate CATE.'
            return y_pred_nji.mean(axis=0)
        elif quantity == 'ATE':
            return y_pred_nji.mean(axis=0)
        else:
            return y_pred_nji

    def comp_transormer(self, x, categories='auto'):
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
            # if not self._is_fitted:
            if not hasattr(self, 'ord_transformer'):
                self.ord_transformer = OrdinalEncoder(categories=categories)
                self.ord_transformer.fit(x)

                labels = [
                    np.arange(len(c)) for c in self.ord_transformer.categories_
                ]
                labels = cartesian(labels)
                categories = [np.arange(len(labels))]

                self.label_dict = {tuple(k): i for i, k in enumerate(labels)}

            x_transformed = self.ord_transformer.transform(x).astype(int)
            x = np.full((x.shape[0], 1), np.NaN)

            for i, x_i in enumerate(x_transformed):
                x[i] = self.label_dict[tuple(x_i)]

        # if not self._is_fitted:
        if not hasattr(self, 'oh_transformer'):
            self.oh_transformer = OneHotEncoder(categories=categories)
            self.oh_transformer.fit(x)

        x = self.oh_transformer.transform(x).toarray()

        return x

    def _effect_nji_all(self, data=None):
        y_nji = self._prepare4est(data=data, all_tr=True)
        return y_nji

    def effect_nji(self, data=None):
        y_nji = self._prepare4est(data=data, all_tr=False).reshape(-1, self._y_d, 1)
        null_effect = np.zeros_like(y_nji)
        
        if self.treat > self.control:
            effect_ = np.concatenate((null_effect, y_nji), axis=2)
        else:
            effect_ = np.concatenate((y_nji, null_effect), axis=2)
        
        return effect_

    def _prepare4est(self, data=None, all_tr=False, treat=None):
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

        if all_tr:
            return y_pred_nji
        else:
            if treat is None:
                return y_pred_nji[:, :, self.treat]
            else:
                treat = get_tr_ctrl(
                    treat,
                    self.comp_transormer,
                    treat=True,
                    one_hot=False,
                    discrete_treat=True,
                )
                return y_pred_nji[:, :, treat]

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
        control,
        treat,
    ):
        final_result = defaultdict(list)
        final_result['is_fitted'].append(False)

        control = get_tr_ctrl(
            control,
            self.comp_transormer,
            treat=False,
            one_hot=False,
            discrete_treat=True,
        )
        treat = get_tr_ctrl(
            treat,
            self.comp_transormer,
            treat=True,
            one_hot=False,
            discrete_treat=True,
        )
        self.treat = treat
        self.control = control

        y_prime_control = (y_prime[:, :, control]).reshape(-1, self._y_d, 1)
        y_prime = y_prime - y_prime_control

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
        """Generate a model for fitting the outcome model for the doubly robust
        model.

        Parameters
        ----------
        model : estimator
            Any valid model should implement the fit() and predict() methods.

        Returns
        -------
        instance of _YModelWrapper
            A wrapped model.
        """
        y_model = clone(model)
        return _YModelWrapper(y_model=y_model, x_d=self._x_d, y_d=self._y_d)


class _DoublyRobustOld(BaseEstModel):
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
