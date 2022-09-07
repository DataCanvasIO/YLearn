import numpy as np

from sklearn import clone
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from collections import defaultdict

from .base_models import BaseEstModel
from .utils import (
    convert2array,
    get_groups,
    get_treat_control,
    get_wv,
    cartesian,
    get_tr_ctrl,
    check_classes,
)


class SLearner(BaseEstModel):
    r"""
    SLearn uses one machine learning model to compute the causal effects.
    Specifically, we fit a model to predict outcome (y) from treatment (x) and
    adjustment (w):

    .. math::

        y = f(x, w).

    The causal effect is then calculated as

    .. math::

        \text{causal_effect} = f(x=1, w) - f(x=0, w).

    Attributes
    ----------
    model : estimator, optional
        The machine learning model for the control group data. Any valid x0_model
        should implement the fit() and predict() methods.

    random_state : int. Defaults to 2022.

    _is_fitted : bool. Defaults to False.
        True if the TLearner is fitted ortherwise False.

    _y_d : int

    treatment : list of str, optional
        Names of the treatments.

    outcome : list of str, optional
        Names of the outcomes.

    adjustment : list of str, optional
        Names of the adjustment set.

    covariate : list of str, optional
        Names of the covariates.

    combined_treat : bool
        Whether use the combined treat technique when training the model.

    categories : str, optional. Defaults to 'auto'.

    combined_treat : bool. Defaults to True.
        Whether combine multiple treatments into a single treatment.

    _is_fitted : bool
        True if the model is fitted ortherwise False.

    Methods
    ----------
    fit(data, outcome, treatment, adjustment, covariate, treat, control, combined_treatment)

    estimate(data, quantity=None)

    _fit_combined_treat(x, wv, y, treat, control, categories, **kwargs)
        Fit function when combined_treat is set to True.

    _comp_transformer(x, categories='auto')
        Transform the discrete treatment into one-hot vectors when combined_treat
        is set to True.

    _fit_separate_treat(x, wv, y, categories)
        Fit function when combined_treat is set to False.

    _prepare4est(data, outcome, treatment, adjustment, individual=None)
        Prepare (fit the model) for estimating various quantities including
        ATE, CATE, ITE, and CITE.

    _prepare_combined_treat(wv)

    _prepare_separate_treat(wv)
    """

    def __init__(
        self,
        model,
        random_state=2022,
        is_discrete_treatment=True,
        is_discrete_outcome=False,
        categories="auto",
        proba_output=False,
        *args,
        **kwargs,
    ):
        """
        Parameters
        ----------
        model : estimator, optional
            The base machine learning model for training SLearner. Any model
            should be some valid machine learning model with fit() and
            predict() functions.

        random_state : int

        categories : str
        """
        assert is_discrete_treatment is True
        if proba_output:
            assert (
                is_discrete_outcome
            ), f"proba_output requires is_discrete_outcome to be True but was given {is_discrete_outcome}"
            assert hasattr(
                model, "predict_proba"
            ), f"The predict_proba method of {model} is required to use proba_output. But None was given."
        self.proba_output = proba_output
        self.pred_func = "predict_proba" if proba_output else "predict"

        self.model = clone(model)
        self._is_fitted = False

        super().__init__(
            random_state=random_state,
            is_discrete_treatment=is_discrete_treatment,
            is_discrete_outcome=is_discrete_outcome,
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
        **kwargs,
    ):
        """Fit the SLearner in the dataset.

        Parameters
        ----------
        data : pandas.DataFrame
            Training dataset for training the estimator.

        outcome : list of str, optional.
            List of names of the outcome features.

        treatment : list of str, optional
            List of names of the treatment features.

        adjustment : List of str, optional
            Lisf ot names of adjustment set ensuring the unconfoundness, by default None

        covariate : List of str, optional
            Covariate features, by default None

        treat : int, optional
            Label of the intended treatment group, by default None

        control : int, optional
            Label of the intended control group, by default None

        combined_treatment : bool, optional
            Only modify this parameter for multiple treatments, where multiple discrete
            treatments are combined to give a single new group of discrete treatment if
            set as True.
            When combined_treatment is set to True, then if there are multiple
            treatments, we can use the combined_treatment technique to covert
            the multiple discrete classification tasks into a single discrete
            classification task. For an example, if there are two different
            binary treatments:
                treatment_1: x_1 | x_1 \in {'sleep', 'run'},
                treatment_2: x_2 | x_2 \in {'study', 'work'},
            then we can convert these two binary classification tasks into
            a single classification task with 4 different classes:
                treatment: x | x \in {0, 1, 2, 3},
            where, for example, 1 stands for ('sleep' and 'stuy').

        Returns
        -------
        instance of SLearner
        """
        assert (
            adjustment is not None or covariate is not None
        ), "Need adjustment set or covariates to perform estimation."

        super().fit(
            data,
            outcome,
            treatment,
            adjustment=adjustment,
            covariate=covariate,
        )

        self.combined_treat = combined_treatment

        # get numpy data
        y, x, w, v = convert2array(data, outcome, treatment, adjustment, covariate)
        self._y_d = y.shape[1]

        # get categories for treatment transformer
        if self.categories == "auto" or self.categories is None:
            categories = "auto"
        else:
            categories = list(self.categories)

        # self.transformer = OneHotEncoder(categories=categories)
        wv = get_wv(w, v)
        self._w = w
        self._wv = wv

        if combined_treatment:
            return self._fit_combined_treat(
                x, wv, y, treat, control, categories, **kwargs
            )
        else:
            return self._fit_separate_treat(x, wv, y, categories, **kwargs)

    def _prepare4est(self, data=None, target_outcome=None):
        if not self._is_fitted:
            raise Exception("The estimator has not been fitted yet.")

        if data is None:
            wv = self._wv
        else:
            w, v = convert2array(data, self.adjustment, self.covariate)
            # if w is None:
            #     w = self._w

            wv = get_wv(w, v)

        if self.combined_treat:
            return self._prepare_combined_treat(wv, target_outcome=target_outcome)
        else:
            return self._prepare_separate_treat(wv, target_outcome=target_outcome)

    def estimate(
        self,
        data=None,
        quantity=None,
        target_outcome=None,
        **kwargs,
    ):
        """Estimate the causal effect with the type of the quantity.

        Parameters
        ----------
        data : pandas.DataFrame, optional. Defaults to None.
            If None, then the training data is used as data.

        quantity : str, optional. Defaults to None
            Option for returned estimation result. The possible values of quantity include:

                1. *'CATE'* : the estimator will evaluate the CATE;

                2. *'ATE'* : the estimator will evaluate the ATE;

                3. *None* : the estimator will evaluate the ITE or CITE.

        Returns
        -------
        ndarray
        """
        if target_outcome is not None:
            assert (
                self.proba_output
            ), f"target_outcome can only be specificed when proba_output is True."

            target_outcome = check_classes(target_outcome, self.model.classes_)

        effect = self._prepare4est(data, target_outcome=target_outcome, **kwargs)
        if quantity == "CATE" or quantity == "ATE":
            return np.mean(effect, axis=0)
        else:
            return effect

    def effect_nji(self, data=None):
        y_nji = self._prepare4est(data=data)

        if y_nji.ndim == 3:
            n, y_d, x_d = y_nji.shape
        elif y_nji.ndim == 2:
            n, y_d = y_nji.shape
            x_d = 1
        else:
            n, y_d, x_d = y_nji.shape[0], 1, 1

        y_nji = y_nji.reshape(n, y_d, x_d)
        zeros_ = np.zeros((n, y_d, 1))
        y_nji = np.concatenate((zeros_, y_nji), axis=2)

        return y_nji

    def _fit_combined_treat(self, x, wv, y, treat, control, categories, **kwargs):
        """Fit function which is used when multiple treatments are combined to
        give a single equivalent treatment vector.

        Parameters
        ----------
        x : np.array
            Treatment variables with shape (n, x_d)

        wv : np.array
            Covariate variables with shape (n, wv_d)

        y : np.array
            Outcome vevariablesctor with shape (n, y_d)

        treat : int or list, optional
            If there is only one treament, then treat indicates the treatment
            group. If there are multiple treatment groups, then treat should
            be a list of str with length equal to the number of treatments.
            For example, when there are multiple
            discrete treatments, array(['run', 'read']) means the treat value of
            the first treatment is taken as 'run' and that of the second treatment
            is taken as 'read'.

        control : int or list, optional
            See treat for more information

        categories : str

        Returns
        -------
        instance of SLearner
        """
        # Converting treatment to array with shape (n, num_treatments)
        # self.transformer = OrdinalEncoder(categories=categories)
        # self.transformer.fit(x)
        # x = self.transformer.transform(x)

        # # For multiple treatments: divide the data into several groups
        # group_categories = self.transformer.categories_
        # n_treatments = len(group_categories)

        x = self._comp_transformer(x, categories=categories)

        self._x_d = x.shape[1]
        self.treat = treat
        self.control = control

        x = np.concatenate((wv, x), axis=1)
        y = y.squeeze()

        self.model.fit(x, y, **kwargs)

        self._is_fitted = True

        return self

    @property
    def outcome_classes_(self):
        if self.is_discrete_outcome:
            assert self._is_fitted, "The model has not been fitted yet."
            return self.model.classes_
        else:
            return None

    def _comp_transformer(self, x, categories="auto"):
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

            for i, x_i in enumerate(x_transformed):
                x[i] = self.label_dict[tuple(x_i)]

        if not self._is_fitted:
            self.oh_transformer = OneHotEncoder(categories=categories)
            self.oh_transformer.fit(x)

        x = self.oh_transformer.transform(x).toarray()

        return x

    def _fit_separate_treat(self, x, wv, y, categories, **kwargs):
        """Fit function which is used when multiple treatments are treated separately,
        i.e., when combined_treat is False.

        Parameters
        ----------
        x : ndarray
            Treatment variables.

        wv : ndarray
            Covariate variables.

        y : ndarray
            Outcome variables.

        categories : str, optional. Defaults to 'auto'.

        Returns
        -------
        instance of SLearner
        """
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

    def _prepare_combined_treat(self, wv, target_outcome=None):
        n = wv.shape[0]
        self.treat = get_tr_ctrl(
            self.treat,
            self._comp_transformer,
            treat=True,
            one_hot=False,
            discrete_treat=True,
        )
        self.control = get_tr_ctrl(
            self.control,
            self._comp_transformer,
            treat=False,
            one_hot=False,
            discrete_treat=True,
        )

        treat = np.zeros(
            self._x_d,
        )
        control = np.zeros(
            self._x_d,
        )

        treat[self.treat] = 1
        control[self.control] = 1

        xt = np.repeat(treat.reshape(1, -1), n, axis=0)
        x0 = np.repeat(control.reshape(1, -1), n, axis=0)

        if len(xt.shape) == 1:
            xt = xt.reshape(-1, 1)
            x0 = x0.reshape(-1, 1)

        xt = np.concatenate((wv, xt), axis=1)
        x0 = np.concatenate((wv, x0), axis=1)

        yt = self.model.__getattribute__(self.pred_func)(xt)
        y0 = self.model.__getattribute__(self.pred_func)(x0)
        if target_outcome is not None:
            yt, y0 = yt[:, target_outcome], y0[:, target_outcome]
        return yt - y0

    def _prepare_separate_treat(self, wv, target_outcome=None):
        n = wv.shape[0]
        x_control = np.zeros((1, self._x_d))
        x_control[:, 0] = 1
        x_control = np.repeat(x_control, n, axis=0).astype(int)
        x_control = np.concatenate((wv, x_control), axis=1)

        if self.proba_output:
            f_nj0 = self.model.predict_proba(x_control)
            y_d = f_nj0.shape[1]
            f_nji = np.full((n, y_d, self._x_d - 1), np.NaN)

            for i in range(self._x_d - 1):
                x_treat = np.zeros((1, self._x_d))
                x_treat[:, i + 1] = 1
                x_treat = np.repeat(x_treat, n, axis=0).astype(int)
                x_treat = np.concatenate((wv, x_treat), axis=1)
                fnji = (self.model.predict_proba(x_treat) - f_nj0).reshape(n, y_d)
                f_nji[:, :, i] = fnji

            if target_outcome is not None:
                f_nji = f_nji[:, target_outcome, :]
        else:
            f_nj0 = self.model.predict(x_control)
            f_nji = np.full((n, self._y_d, self._x_d - 1), np.NaN)

            for i in range(self._x_d - 1):
                x_treat = np.zeros((1, self._x_d))
                x_treat[:, i + 1] = 1
                x_treat = np.repeat(x_treat, n, axis=0).astype(int)
                x_treat = np.concatenate((wv, x_treat), axis=1)
                fnji = (self.model.predict(x_treat) - f_nj0).reshape(n, self._y_d)
                f_nji[:, :, i] = fnji

        return f_nji.squeeze()

    # def __repr__(self) -> str:
    #     return f'SLearner'


class TLearner(BaseEstModel):
    """
    TLearner uses two machine learning models to estimate the causal
    effect. Specifically, we
    1. fit two models for the treatment group (x=treat) and control group
    (x=control), respectively:
        y1 = x1_model(w) with data where x=treat,
        y0 = x0_model(w) with data where x=control;
    2. compute the causal effect as the difference between these two models:
        causal_effect = x1_model(w) - x0_model(w).

    Attributes
    -----------
    xt_model : estimator, optional
        The machine learning model for the treatment group data. Any valid xt_model
        should implement the fit() and predict() methods.

    x0_model : estimator, optional
        The machine learning model for the control group data. Any valid x0_model
        should implement the fit() and predict() methods.

    random_state : int. Defaults to 2022.

    _is_fitted : bool. Defaults to False.
        True if the TLearner is fitted ortherwise False.

    _y_d : int

    transformer : OrdinalEncoder

    categories : str

    treatment : list of str, optional
        Names of the treatments.

    outcome : list of str, optional
        Names of the outcomes.

    adjustment : list of str, optional
        Names of the adjustment set.

    covariate : list of str, optional
        Names of the covariates.

    combined_treat : bool
        Whether use the combined treat technique when training the model.

    _wv : ndarray with shape (n, w_d + v_d)

    _w : ndarray with shape (n, w_d)

    Methods
    ----------
    fit(data, outcome, treatment, adjustment, covariate, treat, control, combined_treatment)

    estimate(data, quantity=None)

    _fit_combined_treat(x, wv, y, treat, control, categories, **kwargs)
        Fit function when combined_treat is set to True.

    _comp_transformer(x, categories='auto')
        Transform the discrete treatment into one-hot vectors when combined_treat
        is set to True.

    _fit_separate_treat(x, wv, y, categories)
        Fit function when combined_treat is set to False.

    _prepare4est(data, outcome, treatment, adjustment, individual=None)
        Prepare (fit the model) for estimating various quantities including
        ATE, CATE, ITE, and CITE.

    _prepare_combined_treat(wv)

    _prepare_separate_treat(wv)
    """

    def __init__(
        self,
        model,
        random_state=2022,
        is_discrete_treatment=True,
        is_discrete_outcome=False,
        proba_output=False,
        categories="auto",
        **kwargs,
    ):
        """

        Parameters
        ----------
        model : estimator, optional
            The base machine learning model for training TLearner. Any model
            should be some valid machine learning model with fit() and
            predict() functions.

        random_state : int

        categories : str
        """
        assert is_discrete_treatment is True

        if proba_output:
            assert (
                is_discrete_outcome
            ), f"proba_output requires is_discrete_outcome to be True but was given {is_discrete_outcome}"
            assert hasattr(
                model, "predict_proba"
            ), f"The predict_proba method of {model} is required to use proba_output. But None was given."

        self.proba_output = proba_output
        self.pred_func = "predict_proba" if proba_output else "predict"

        self.model = clone(model)
        self.xt_model = clone(model)
        self.x0_model = clone(model)
        self._is_fitted = False

        super().__init__(
            random_state=random_state,
            is_discrete_treatment=is_discrete_treatment,
            is_discrete_outcome=is_discrete_outcome,
            categories=categories,
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
        **kwargs,
    ):
        """Fit the TLearner in the dataset.

        Parameters
        ----------
        data : pandas.DataFrame
            Training dataset for training the estimator.

        outcome : list of str, optional.
            List of names of the outcome features.

        treatment : list of str, optional
            List of names of the treatment features.

        adjustment : List of str, optional
            Lisf ot names of adjustment set ensuring the unconfoundness, by default None

        covariate : List of str, optional
            Covariate features, by default None

        treat : int, optional
            Label of the intended treatment group, by default None

        control : int, optional
            Label of the intended control group, by default None

        combined_treatment : bool, optional
            Only modify this parameter for multiple treatments, where multiple discrete
            treatments are combined to give a single new group of discrete treatment if
            set as True.
            When combined_treatment is set to True, then if there are multiple
            treatments, we can use the combined_treatment technique to covert
            the multiple discrete classification tasks into a single discrete
            classification task. For an example, if there are two different
            binary treatments:
                treatment_1: x_1 | x_1 \in {'sleep', 'run'},
                treatment_2: x_2 | x_2 \in {'study', 'work'},
            then we can convert these two binary classification tasks into
            a single classification task with 4 different classes:
                treatment: x | x \in {0, 1, 2, 3},
            where, for example, 1 stands for ('sleep' and 'stuy').

        Returns
        -------
        instance of TLearner
        """
        assert (
            adjustment is not None or covariate is not None
        ), "Need adjustment set or covariates to perform estimation."
        super().fit(
            data,
            outcome,
            treatment,
            adjustment=adjustment,
            covariate=covariate,
        )

        self.combined_treat = combined_treatment

        y, x, w, v = convert2array(data, outcome, treatment, adjustment, covariate)
        self._y_d = y.shape[1]

        if self.categories == "auto" or self.categories is None:
            categories = "auto"
        else:
            categories = list(self.categories)

        # self.transformer = OrdinalEncoder(categories=categories)
        # self.transformer.fit(x)
        # x = self.transformer.transform(x)

        # self.group_categories = self.transformer.categories_
        wv = get_wv(w, v)
        self._wv = wv
        self._w = w

        if combined_treatment:
            return self._fit_combined_treat(
                x, wv, y, treat, control, categories, **kwargs
            )
        else:
            return self._fit_separate_treat(x, wv, y, categories, **kwargs)

    def estimate(
        self,
        data=None,
        quantity=None,
        target_outcome=None,
        **kwargs,
    ):
        """Estimate the causal effect with the type of the quantity.

        Parameters
        ----------
        data : pandas.DataFrame, optional. Defaults to None.
            If None, then the training data is used as data.

        quantity : str, optional. Defaults to None
            The possible values of quantity include:
                'CATE' : the estimator will evaluate the CATE;
                'ATE' : the estimator will evaluate the ATE;
                None : the estimator will evaluate the ITE or CITE.

        Returns
        -------
        ndarray
        """
        if target_outcome is not None:
            assert (
                self.proba_output
            ), f"target_outcome can only be specificed when proba_output is True."

        target_outcome = check_classes(target_outcome, self.outcome_classes_)

        effect = self._prepare4est(data, target_outcome=target_outcome)
        if quantity == "CATE" or quantity == "ATE":
            return np.mean(effect, axis=0)
        else:
            return effect

    @property
    def outcome_classes_(self):
        if self.is_discrete_outcome:
            assert self._is_fitted, "The model has not been fitted yet."
            try:
                classes_ = self.xt_model.classes_
            except:
                classes_ = self._fitted_dict_separa["models"][0].classes_
        else:
            classes_ = None

        return classes_

    def effect_nji(self, data=None):
        y_nji = self._prepare4est(data=data)

        if y_nji.ndim == 3:
            n, y_d, x_d = y_nji.shape
        elif y_nji.ndim == 2:
            n, y_d = y_nji.shape
            x_d = 1
        else:
            n, y_d, x_d = y_nji.shape[0], 1, 1

        y_nji = y_nji.reshape(n, y_d, x_d)
        zeros_ = np.zeros((n, y_d, 1))
        y_nji = np.concatenate((zeros_, y_nji), axis=2)

        return y_nji

    def _prepare4est(self, data=None, target_outcome=None):
        if not self._is_fitted:
            raise Exception("The estimator has not been fitted yet.")

        if data is None:
            wv = self._wv
        else:
            w, v = convert2array(data, self.adjustment, self.covariate)
            # if w is None:
            #     w = self._w
            wv = get_wv(w, v)

        if self.combined_treat:
            return self._prepare_combined_treat(wv, target_outcome=target_outcome)
        else:
            return self._prepare_separate_treat(wv, target_outcome=target_outcome)

    def _fit_combined_treat(self, x, wv, y, treat, control, categories, **kwargs):
        """Fit function which is used when multiple treatments are combined to
        give a single equivalent treatment vector.

        Parameters
        ----------
        x : np.array
            Treatment variables with shape (n, x_d)

        wv : np.array
            Covariate variables with shape (n, wv_d)

        y : np.array
            Outcome vevariablesctor with shape (n, y_d)

        treat : int or ndarray, optional
            If there is only one treament, then treat indicates the treatment
            group. If there are multiple treatment groups, then treat should
            be an ndarray of str with length equal to the number of treatments.
            For example, when there are multiple
            discrete treatments, array(['run', 'read']) means the treat value of
            the first treatment is taken as 'run' and that of the second treatment
            is taken as 'read'.

        control : int or list, optional
            See treat for more information

        categories : str

        Returns
        -------
        instance of TLearner
        """
        # specify the treat and control label
        x = self._comp_transformer(x, categories=categories)

        treat = get_tr_ctrl(
            treat,
            self._comp_transformer,
            treat=True,
            one_hot=False,
            discrete_treat=True,
        )
        control = get_tr_ctrl(
            control,
            self._comp_transformer,
            treat=False,
            one_hot=False,
            discrete_treat=True,
        )
        # treat = get_treat_control(treat, n_treatments, True)
        # control = get_treat_control(control, n_treatments, False)
        self.treat = treat
        self.control = control

        wv_treat, y_treat = get_groups(treat, x, True, wv, y)
        wv_control, y_control = get_groups(control, x, True, wv, y)

        y_treat = y_treat.squeeze()
        y_control = y_control.squeeze()

        self.xt_model.fit(wv_treat, y_treat, **kwargs)
        self.x0_model.fit(wv_control, y_control, **kwargs)

        self._is_fitted = True

        return self

    def _comp_transformer(self, x, categories="auto"):
        """Transform the discrete treatment into one-hot vectors.

        Parameters
        ----------
        x : ndarray, shape (n, x_d)
            An array containing the information of the treatment variables

        categories : str or list, optional
            by default 'auto

        Returns
        -------
        ndarray
            The transformed one-hot vectors
        """
        if x.shape[1] > 1:
            # if not self._is_fitted:
            if not hasattr(self, "ord_transformer"):
                self.ord_transformer = OrdinalEncoder(categories=categories)
                self.ord_transformer.fit(x)

                self.group_categories = self.ord_transformer.categories_

                labels = [np.arange(len(c)) for c in self.group_categories]
                labels = cartesian(labels)
                categories = [np.arange(len(labels))]

                self.label_dict = {tuple(k): i for i, k in enumerate(labels)}

            x_transformed = self.ord_transformer.transform(x).astype(int)
            x = np.full((x.shape[0], 1), np.NaN)

            for i, x_i in enumerate(x_transformed):
                x[i] = self.label_dict[tuple(x_i)]

        if not hasattr(self, "oh_transformer"):
            self.oh_transformer = OneHotEncoder()
            self.oh_transformer.fit(x)

        x = self.oh_transformer.transform(x).toarray()

        return x

    def _fit_separate_treat(self, x, wv, y, categories, **kwargs):
        """Fit function which is used when multiple treatments are treated separately,
        i.e., when combined_treat is False. For example, if there are 5 different
        discrete treatments, then we will fit 5 different models for them, respectivley.

        Parameters
        ----------
        x : ndarray
            Treatment variables.

        wv : ndarray
            Covariate variables.

        y : ndarray
            Outcome variables.

        group_categories : ndarray of ndarray
            Each ndarray in the group_categories is the classes of a single
            treatment vector. For example,
                array([array(['good', 'bad']), array([array(['a', 'b'])])])
            means there are two treatment vectors where the first treatment
            vector has classes 'good' and 'bad' while the second treatment
            vector has classes 'a' and 'b'.

        Returns
        -------
        instance of TLearner
        """
        self.transformer = OrdinalEncoder(categories=categories)
        self.transformer.fit(x)
        x = self.transformer.transform(x)

        self.group_categories = self.transformer.categories_

        self._fitted_dict_separa = defaultdict(list)
        waited_treat = [np.arange(len(i)) for i in self.group_categories]
        treat_arrays = cartesian(waited_treat)

        for treat in treat_arrays:
            model = clone(self.xt_model)
            _wv, _y = get_groups(treat, x, False, wv, y)
            _y = _y.squeeze()

            model.fit(_wv, _y, **kwargs)
            self._fitted_dict_separa["treatment"].append(treat)
            self._fitted_dict_separa["models"].append(model)

        self._is_fitted = True

        return self

    def _prepare_combined_treat(self, wv, target_outcome=None):
        yt = self.xt_model.__getattribute__(self.pred_func)(wv)
        y0 = self.x0_model.__getattribute__(self.pred_func)(wv)
        if target_outcome is not None:
            yt, y0 = yt[:, target_outcome], y0[:, target_outcome]
        return yt - y0

    def _prepare_separate_treat(self, wv, target_outcome=None):
        n_treatments = len(self._fitted_dict_separa["treatment"])
        n = wv.shape[0]
        if self.proba_output:
            f_nj0 = self._fitted_dict_separa["models"][0].predict_proba(wv)
            y_d = f_nj0.shape[1]
            f_nji = np.full((n, y_d, n_treatments - 1), np.NaN)
            for i, model in enumerate(self._fitted_dict_separa["models"][1:]):
                fnji = (model.predict_proba(wv) - f_nj0).reshape(n, y_d)
                f_nji[:, :, i] = fnji
            if target_outcome is not None:
                f_nji = f_nji[:, target_outcome, :]
        else:
            f_nji = np.full((n, self._y_d, n_treatments - 1), np.NaN)
            f_nj0 = self._fitted_dict_separa["models"][0].predict(wv)

            for i, model in enumerate(self._fitted_dict_separa["models"][1:]):
                fnji = (model.predict(wv) - f_nj0).reshape(n, self._y_d)
                f_nji[:, :, i] = fnji

        return f_nji.squeeze()

    # def __repr__(self) -> str:
    #     return f'TLearner'


class XLearner(BaseEstModel):
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
    ft_model : estiamtor, optional
        Machine learning model for the treatment gropu in the step 1.

    f0_model : estiamtor, optional
        Machine learning model for the control gropu in the step 1.

    kt_model : estiamtor, optional
        Machine learning model for the treatment gropu in the step 2.

    k0_model : estiamtor, optional
        Machine learning model for the control gropu in the step 2.

    _is_fitted : bool
        True if the instance of XLearner is fitted otherwise False.

    categories : str

    treatment : list of str, optional
        Names of the treatments.

    outcome : list of str, optional
        Names of the outcomes.

    adjustment : list of str, optional
        Names of the adjustment set.

    covariate : list of str, optional
        Names of the covariates.

    combined_treat : bool
        Whether use the combined treat technique when training the model.

    Methods
    ----------
    fit(data, outcome, treatment, adjustment, covariate, treat, control, combined_treatment)

    estimate(data, quantity=None)

    _fit_combined_treat(x, wv, y, treat, control, categories, **kwargs)
        Fit function when combined_treat is set to True.

    _comp_transformer(x, categories='auto')
        Transform the discrete treatment into one-hot vectors when combined_treat
        is set to True.

    _fit_separate_treat(x, wv, y, categories)
        Fit function when combined_treat is set to False.

    _prepare4est(data, outcome, treatment, adjustment, individual=None)
        Prepare (fit the model) for estimating various quantities including
        ATE, CATE, ITE, and CITE.

    _prepare_combined_treat(wv)

    _prepare_separate_treat(wv)
    """

    def __init__(
        self,
        model,
        random_state=2022,
        is_discrete_treatment=True,
        is_discrete_outcome=False,
        proba_output=False,
        final_proba_model=None,
        categories="auto",
        **kwargs,
    ):
        """
        Parameters
        ----------
        model : estimator, optional
            The base machine learning model for training XLearner. Any model
            should be some valid machine learning model with fit() and
            predict() functions.

        random_state : int

        categories : str
        """
        assert is_discrete_treatment is True

        final_effect_model = clone(model)

        if proba_output:
            assert (
                is_discrete_outcome
            ), f"proba_output requires is_discrete_outcome to be True but was given {is_discrete_outcome}"
            assert hasattr(
                model, "predict_proba"
            ), f"The predict_proba method of {model} is required to use proba_output."
            if final_proba_model is not None:
                final_effect_model = clone(final_proba_model)

        self.proba_output = proba_output
        self.pred_func = "predict_proba" if proba_output else "predict"

        self.model = clone(model)
        self.ft_model = clone(model)
        self.f0_model = clone(model)
        self.kt_model = clone(final_effect_model)
        self.k0_model = clone(final_effect_model)

        self._is_fitted = False

        super().__init__(
            random_state=random_state,
            is_discrete_treatment=is_discrete_treatment,
            is_discrete_outcome=is_discrete_outcome,
            categories=categories,
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
        **kwargs,
    ):
        """Fit the XLearner in the dataset.

        Parameters
        ----------
        data : pandas.DataFrame
            Training dataset for training the estimator.

        outcome : list of str, optional.
            List of names of the outcome features.

        treatment : list of str, optional
            List of names of the treatment features.

        adjustment : List of str, optional
            Lisf ot names of adjustment set ensuring the unconfoundness, by default None

        covariate : List of str, optional
            Covariate features, by default None

        treat : int, optional
            Label of the intended treatment group, by default None

        control : int, optional
            Label of the intended control group, by default None

        combined_treatment : bool, optional
            Only modify this parameter for multiple treatments, where multiple discrete
            treatments are combined to give a single new group of discrete treatment if
            set as True.
            When combined_treatment is set to True, then if there are multiple
            treatments, we can use the combined_treatment technique to covert
            the multiple discrete classification tasks into a single discrete
            classification task. For an example, if there are two different
            binary treatments:
                treatment_1: x_1 | x_1 \in {'sleep', 'run'},
                treatment_2: x_2 | x_2 \in {'study', 'work'},
            then we can convert these two binary classification tasks into
            a single classification task with 4 different classes:
                treatment: x | x \in {0, 1, 2, 3},
            where, for example, 1 stands for ('sleep' and 'stuy').

        Returns
        -------
        instance of XLearner
        """
        assert (
            adjustment is not None or covariate is not None
        ), "Need adjustment set or covariates to perform estimation."

        super().fit(
            data,
            outcome,
            treatment,
            adjustment=adjustment,
            covariate=covariate,
        )

        self.combined_treat = combined_treatment

        y, x, w, v = convert2array(data, outcome, treatment, adjustment, covariate)
        self._y_d = y.shape[1]

        if self.categories == "auto" or self.categories is None:
            categories = "auto"
        else:
            categories = list(self.categories)

        # self.transformer = OrdinalEncoder(categories=categories)
        # self.transformer.fit(x)
        # x = self.transformer.transform(x)

        # self.group_categories = self.transformer.categories_
        wv = get_wv(w, v)
        self._wv = wv
        self._w = w

        if combined_treatment:
            return self._fit_combined_treat(
                x, wv, y, treat, control, categories, **kwargs
            )
        else:
            return self._fit_separate_treat(x, wv, y, categories, **kwargs)

    @property
    def outcome_classes_(self):
        if self.is_discrete_outcome:
            assert self._is_fitted, "The model has not been fitted yet."
            try:
                classes_ = (
                    self.ft_model.classes_
                )  # TODO: note that, in some extreme case when the treat group only has one class of outcome, then this will render an error, but currently we ignore such case
            except:
                classes_ = self._fitted_dict_separa["models"][0].classes_
            # else:
            #     classes_ = None
        else:
            classes_ = None

        return classes_

    def _prepare4est(self, data=None, rho=0.5, target_outcome=None, *args, **kwargs):
        if not self._is_fitted:
            raise Exception("The estimator has not been fitted yet.")

        if data is None:
            wv = self._wv
        else:
            w, v = convert2array(data, self.adjustment, self.covariate)
            # if w is None:
            #     w = self._w
            wv = get_wv(w, v)

        if self.combined_treat:
            effect = self._prepare_combined_treat(
                wv, rho, target_outcome=target_outcome
            )
        else:
            effect = self._prepare_separate_treat(
                wv, rho, target_outcome=target_outcome
            )

        return effect

    def estimate(
        self, data=None, rho=0.5, quantity=None, target_outcome=None, *args, **kwargs
    ):
        if target_outcome is not None:
            assert (
                self.proba_output
            ), f"target_outcome can only be specificed when proba_output is True."

        target_outcome = check_classes(target_outcome, self.outcome_classes_)
        # TODO: add support for other types of rho
        effect = self._prepare4est(
            data, rho, target_outcome=target_outcome, *args, **kwargs
        )

        if quantity == "CATE" or quantity == "ATE":
            return np.mean(effect, axis=0)
        else:
            return effect

    def effect_nji(self, data=None):
        y_nji = self._prepare4est(data=data)

        if y_nji.ndim == 3:
            n, y_d, x_d = y_nji.shape
        elif y_nji.ndim == 2:
            n, y_d = y_nji.shape
            x_d = 1
        else:
            n, y_d, x_d = y_nji.shape[0], 1, 1

        y_nji = y_nji.reshape(n, y_d, x_d)
        zeros_ = np.zeros((n, y_d, 1))
        y_nji = np.concatenate((zeros_, y_nji), axis=2)

        return y_nji

    def _fit_combined_treat(self, x, wv, y, treat, control, categories, **kwargs):
        """Fit function which is used when multiple treatments are combined to
        give a single equivalent treatment vector.

        Parameters
        ----------
        x : np.array
            Treatment variables with shape (n, x_d)

        wv : np.array
            Covariate variables with shape (n, wv_d)

        y : np.array
            Outcome vevariablesctor with shape (n, y_d)

        treat : int or list, optional
            If there is only one treament, then treat indicates the treatment
            group. If there are multiple treatment groups, then treat should
            be a list of str with length equal to the number of treatments.
            For example, when there are multiple
            discrete treatments, array(['run', 'read']) means the treat value of
            the first treatment is taken as 'run' and that of the second treatment
            is taken as 'read'.

        control : int or list, optional
            See treat for more information

        categories : str

        Returns
        -------
        instance of XfLearner
        """
        x = self._comp_transformer(x, categories)

        treat = get_tr_ctrl(
            treat,
            self._comp_transformer,
            treat=True,
            one_hot=False,
            discrete_treat=True,
        )
        control = get_tr_ctrl(
            control,
            self._comp_transformer,
            treat=False,
            one_hot=False,
            discrete_treat=True,
        )
        self.treat = treat
        self.control = control

        # Step 1
        # TODO: modify the generation of group ids for efficiency
        wv_treat, y_treat = get_groups(treat, x, True, wv, y)
        wv_control, y_control = get_groups(control, x, True, wv, y)

        y_treat = y_treat.squeeze()
        y_control = y_control.squeeze()

        self.ft_model.fit(wv_treat, y_treat, **kwargs)
        self.f0_model.fit(wv_control, y_control, **kwargs)

        # Step 2
        if self.proba_output:
            self._outcome_oh = OneHotEncoder(categories=[self.ft_model.classes_])
            y_treat = self._outcome_oh.fit_transform(y_treat.reshape(-1, 1))
            y_control = self._outcome_oh.transform(y_control.reshape(-1, 1))

        h_treat_target = y_treat - self.f0_model.__getattribute__(self.pred_func)(
            wv_treat
        )
        h_control_target = (
            self.ft_model.__getattribute__(self.pred_func)(wv_control) - y_control
        )
        self.kt_model.fit(wv_treat, h_treat_target)
        self.k0_model.fit(wv_control, h_control_target)

        # Step 3
        # This is the task of predict. See _prepare4est

        self._is_fitted = True

        return self

    def _comp_transformer(self, x, categories="auto"):
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
            if not hasattr(self, "ord_transformer"):
                self.ord_transformer = OrdinalEncoder(categories=categories)
                self.ord_transformer.fit(x)

                self.group_categories = self.ord_transformer.categories_

                labels = [np.arange(len(c)) for c in self.group_categories]
                labels = cartesian(labels)
                categories = [np.arange(len(labels))]

                self.label_dict = {tuple(k): i for i, k in enumerate(labels)}

            x_transformed = self.ord_transformer.transform(x).astype(int)
            x = np.full((x.shape[0], 1), np.NaN)

            for i, x_i in enumerate(x_transformed):
                x[i] = self.label_dict[tuple(x_i)]

        if not hasattr(self, "oh_transformer"):
            self.oh_transformer = OneHotEncoder()
            self.oh_transformer.fit(x)

        x = self.oh_transformer.transform(x).toarray()

        return x

    def _fit_separate_treat(self, x, wv, y, categories, *args, **kwargs):
        """Fit function which is used when multiple treatments are treated separately,
        i.e., when combined_treat is False.

        Parameters
        ----------
        x : ndarray
            Treatment variables.

        wv : ndarray
            Covariate variables.

        y : ndarray
            Outcome variables.

        group_categories : ndarray of ndarray
            Each ndarray in the group_categories is the classes of a single
            treatment vector. For example,
                array([array(['good', 'bad']), array([array(['a', 'b'])])])
            means there are two treatment vectors where the first treatment
            vector has classes 'good' and 'bad' while the second treatment
            vector has classes 'a' and 'b'.

        Returns
        -------
        instance of XLearner
        """
        self.transformer = OrdinalEncoder(categories=categories)
        self.transformer.fit(x)
        x = self.transformer.transform(x)

        self.group_categories = self.transformer.categories_

        self._fitted_dict_separa = defaultdict(list)
        waited_treat = [np.arange(len(i)) for i in self.group_categories]
        treat_arrays = cartesian(waited_treat)

        f0_model = clone(self.f0_model)
        _wv_control, _y_control = get_groups(treat_arrays[0], x, False, wv, y)
        _y_control = _y_control.squeeze()
        f0_model.fit(_wv_control, _y_control)

        if self.proba_output:
            self._outcome_oh = OneHotEncoder(categories=[f0_model.classes_])
            _y_control = self._outcome_oh.fit_transform(_y_control.reshape(-1, 1))
            for treat in treat_arrays[1:]:
                ft_model = clone(self.ft_model)
                _wv, _y = get_groups(treat, x, False, wv, y)
                _y = _y.squeeze()

                # Step 1
                ft_model.fit(_wv, _y, *args, **kwargs)

                # Step 2
                _y = self._outcome_oh.transform(_y.reshape(-1, 1))

                h_treat_target = _y - f0_model.predict_proba(_wv)
                h_control_target = ft_model.predict_proba(_wv_control) - _y_control
                kt_model = clone(self.kt_model)
                k0_model = clone(self.k0_model)
                kt_model.fit(_wv, h_treat_target)
                k0_model.fit(_wv_control, h_control_target)

                # Step 3
                self._fitted_dict_separa["models"].append((kt_model, k0_model))
                self._fitted_dict_separa["treatment"].append(treat)
        else:
            for treat in treat_arrays[1:]:
                ft_model = clone(self.ft_model)
                _wv, _y = get_groups(treat, x, False, wv, y)
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
                self._fitted_dict_separa["models"].append((kt_model, k0_model))
                self._fitted_dict_separa["treatment"].append(treat)
        self._is_fitted = True

        return self

    def _prepare_combined_treat(self, wv, rho, target_outcome=None):
        # TODO: add support for training to select rho
        kt_pred = self.kt_model.predict(wv)
        k0_pred = self.k0_model.predict(wv)
        ce = rho * kt_pred + (1 - rho) * k0_pred
        if target_outcome is None:
            return ce
        else:
            return ce[:, target_outcome]

    def _prepare_separate_treat(self, wv, rho, target_outcome=None):
        model_list = self._fitted_dict_separa["models"]
        n_treatments = len(self._fitted_dict_separa["treatment"])
        n = wv.shape[0]
        if self.proba_output:
            y_d = self.outcome_classes_.shape[0]
        else:
            y_d = self._y_d
        f_nji = np.full((n, y_d, n_treatments), np.NaN)

        for i, (kt_model, k0_model) in enumerate(model_list):
            pred_t = kt_model.predict(wv)
            pred_0 = k0_model.predict(wv)
            fnji = (rho * pred_t + (1 - rho) * pred_0).reshape(n, self._y_d)
            f_nji[:, :, i] = fnji

        if target_outcome is None:
            return f_nji.squeeze()
        else:
            return f_nji[:, target_outcome, :].squeeze()

    # def __repr__(self) -> str:
    #     return f'XLearner'
