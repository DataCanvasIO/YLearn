import numpy as np
from sklearn.preprocessing import OneHotEncoder


from .base_models import BaseEstLearner
from .utils import (convert2array, get_wv, convert4onehot,
                                   get_treat_control)


class ApproxBound(BaseEstLearner):
    """
    Parameters
    ----------
    y_model : MLModel, optional.
        Machine learning models for fitting the relation between treatment
        and outcome.
    x_prob : np.array of float, optional. Defaults to None.
        The probability of taking specific treatments.
    x_model : MLModel, optional. Defaults to None.
        Machine learning models for fitting the relation between treatment
        and condition set if condition set is True.
    random_state : int
    is_discrete_treatent : bool
    categories : str or list, optional. Deafaults to 'auto'
    """
    # TODO: may consider multiple treatments

    def __init__(
        self,
        y_model,
        x_prob=None,
        x_model=None,
        random_state=2022,
        is_discrete_treatment=True,
        categories='auto'
    ):
        self.y_model = y_model
        self.x_prob = x_prob
        self.x_model = x_model

        super().__init__(
            random_state,
            is_discrete_treatment,
            categories=categories
        )

    def fit(
        self,
        data,
        outcome,
        treatment,
        covariate=None,
        is_discrete_covariate=False,
        **kwargs
    ):
        """_summary_

        Parameters
        ----------
        data : _type_
            _description_
        outcome : _type_
            _description_
        treatment : _type_
            _description_
        covariate : _type_, optional
            _description_, by default None
        is_discrete_covariate : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        super().fit(
            data, outcome, treatment,
            covariate=covariate,
            is_discrete_covariate=is_discrete_covariate
        )
        y, x, v = convert2array(data, outcome, treatment, covariate)
        self._y_max = y.max(axis=0)
        self._y_min = y.min(axis=0)
        self._n = len(data)

        if not self.is_discrete_treatment:
            raise ValueError(
                'The method only supports discrete treatments currently.'
            )

        self.treatment_transformer = OneHotEncoder()
        x_one_hot = self.treatment_transformer.fit_transform(x).toarray()
        self._num_treatments = len(self.treatment_transformer.categories_)
        self._x_d = x_one_hot.shape[1]

        if self.x_prob is None:
            if v is not None:
                x = convert4onehot(x_one_hot).astype(int)

                if self.is_discrete_covariate:
                    v = OneHotEncoder().fit_transform(v)

                self.x_model.fit(v, x)
                self.x_prob = self.x_model.predic_proba(v)
            else:
                self.x_prob = (
                    data[treatment].value_counts().sort_index() / self._n
                ).values.reshape(1, -1)
        else:
            # TODO: modify the following line for multiple treatment
            self.x_prob = self.x_prob.reshape(1, -1)

        self._v = v
        xv = get_wv(x_one_hot, v)
        self.y_model.fit(xv, y.squeeze(), **kwargs)

        self._is_fitted = True

        return self

    def estimate(
        self,
        data=None,
        treat=None,
        control=None,
        y_upper=None,
        y_lower=None,
        assump=None,
        **kwargs
    ):
        """_summary_

        Parameters
        ----------
        data : _type_, optional
            _description_, by default None
        treat : _type_, optional
            _description_, by default None
        control : _type_, optional
            _description_, by default None
        y_upper : _type_, optional
            _description_, by default None
        y_lower : _type_, optional
            _description_, by default None
        assump : str, optional.  Defaults to 'no-assump'.
            Should be one of
                1. no-assump: calculate the no assumption bound whose result
                    will always contain 0.
                2. non-negative: The treatment is always positive.
                3. non-positive: The treatment is always negative.
                4. optimal: The treatment is taken if its effect is positive.


        Returns
        -------
        _type_
            _description_

        Raises
        ------
        Exception
            _description_
        Exception
            _description_
        """
        if not self._is_fitted:
            raise Exception('The estimator is not fitted yet.')

        normal, optim1, optim2 = self._prepare4est(
            data, treat, control, y_upper, y_lower
        )
        upper, lower = normal

        if assump == 'no-assump' or assump is None:
            return (lower, upper)
        elif assump == 'non-negative':
            return (0, upper)
        elif assump == 'non-positive':
            return (lower, 0)
        elif assump == 'optimal':
            # TODO: need to update this for discrete treatment
            return (optim1, optim2)
        else:
            raise Exception(
                'Only support assumptions in no-assump, non-negative, and'
                'non-positive'
            )

    def _prepare4est(
        self,
        data,
        treat,
        control,
        y_upper,
        y_lower,
    ):
        # x_d = self._x_d
        x_prob = self.x_prob

        if self.covariate is None:
            n = 1
            v = None
            if data is None:
                y_upper = self._y_max if y_upper is None else y_upper
                y_lower = self._y_min if y_lower is None else y_lower
            else:
                y = convert2array(data, self.outcome)
                y_upper = y.max(axis=0) if y_upper is None else y_upper
                y_lower = y.min(axis=0) if y_lower is None else y_lower
        else:
            if data is None:
                v = self._v
                y_upper = self._y_max if y_upper is None else y_upper
                y_lower = self._y_min if y_lower is None else y_lower
                n = self._n
            else:
                y, v = convert2array(data, self.outcome, self.covariate)
                y_upper = y.max(axis=0) if y_upper is None else y_upper
                y_lower = y.min(axis=0) if y_lower is None else y_lower
                n = y.shape[0]
                x_prob = self.x_model.predic_proba(v)

        treat = get_treat_control(treat, self._num_treatments, True)
        control = get_treat_control(control, self._num_treatments, False)

        # TODO: modify the following line for multiple treatment
        xt, x0 = np.zeros((n, self._x_d)), np.zeros((n, self._x_d))
        xt[:, treat] = 1
        x0[:, control] = 1

        xt = get_wv(xt, v)
        x0 = get_wv(xt, v)
        yt = self.y_model.predict(xt)
        y0 = self.y_model.predict(x0)
        xt_prob = x_prob[:, treat]
        x0_prob = x_prob[:, control]

        upper = xt_prob * yt + (1 - xt_prob) * y_upper \
            - (1 - x0_prob) * y_lower - x0_prob * y0
        lower = xt_prob * yt + (1 - xt_prob) * y_lower \
            - (1 - x0_prob) * y_upper - x0_prob * y0

        optimal_upper1 = xt_prob * yt - (1 - x0_prob) * y_lower \
            + (1 - xt_prob - x0_prob) * y_upper
        optimal_lower1 = x0_prob * lower - x0_prob * y0
        optimal_upper2 = yt - xt_prob * lower - x0_prob * y0
        optimal_lower2 = xt_prob * yt + x0_prob * lower - y0

        return ((upper, lower), (optimal_upper1, optimal_lower1),
                (optimal_upper2, optimal_lower2))
