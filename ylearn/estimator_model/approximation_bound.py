import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


from .base_models import BaseEstModel
from .utils import (convert2array, get_wv, convert4onehot,
                    get_tr_ctrl, cartesian, get_groups)


class ApproxBound(BaseEstModel):
    """
    A model used for estimating the upper and lower bounds of the causal effects.
    
    Attributes
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
    
    Methods
    ----------
    fit(data, outcome, treatment, covariate=None, is_discrete_covariate=False,
        **kwargs)
    
    estimate(data=None, treat=None, control=None, y_upper=None, y_lower=None, assump=None,)
        Estimate the approximation bound of the causal effect of the treatment
        on the outcome.
    
    comp_transformer(x, categories='auto')
        Transform the discrete treatment into one-hot vectors.
    
    _prepare4est(data, treat, control, y_upper, y_lower,)
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
        """
        Parameters
        ----------
        y_model : estimator, optional
            Any valid y_model should implement the fit() and predict() methods
        
        x_prob : ndarray of shape (c, ), optional. Default to None
            An array of probabilities assigning to the corresponding values of x
            where c is the number of different treatment classes. All elements
            in the array are positive and sumed to 1. For example, x_prob = 
            array([0.5, 0.5]) means both x = 0 and x = 1 take probability 0.5.
            Please set this as None if you are using multiple treatments.
        
        x_model : estimator, optional. Default to None
            Models for predicting the probabilities of treatment. Any valid x_model should implement the fit() and predict_proba() methods.
        
        random_state : int, optional. Defaults to 2022.
        
        is_discrete_treatment : bool, optional
            True if the treatment is discrete.
        
        categories : str, optional
            _description_, by default 'auto'
        """
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
        """Fit x_model and y_model.

        Parameters
        ----------
        data : pandas.DataFrame
            Training data.
        
        outcome : list of str, optional
            Names of the outcome.
        
        treatment : list of str, optional
            Names of the treatment.
        
        covariate : list of str, optional. Defaults to None
            Names of the covariate.
        
        is_discrete_covariate : bool, optional. Defaults to False.

        Returns
        -------
        instance of ApproxBound
            The fitted instance of ApproxBound.

        Raises
        ------
        ValueError
            Raise error when the treatment is not discrete.
        """
        super().fit(
            data, outcome, treatment,
            covariate=covariate,
            is_discrete_covariate=is_discrete_covariate
        )

        y, x, v = convert2array(data, outcome, treatment, covariate)
        # self._y_max = y.max(axis=0)
        # self._y_min = y.min(axis=0)
        self._y = y
        self._n = len(data)

        if not self.is_discrete_treatment:
            raise ValueError(
                'The method only supports discrete treatments currently.'
            )

        if self.categories == 'auto' or self.categories is None:
            categories = 'auto'
        else:
            categories = list(self.categories)

        x_one_hot = self.comp_transormer(x, categories=categories)
        # self._num_treatments = len(self.treatment_transformer.categories_)
        self._x_d = x_one_hot.shape[1]

        self.x_label = np.array(convert4onehot(x_one_hot).astype(int))

        if self.x_prob is None:
            if v is not None:
                if self.is_discrete_covariate:
                    v = OneHotEncoder().fit_transform(v)

                self.x_model.fit(v, self.x_label)
                self.x_prob = self.x_model.predict_proba(v)
            else:
                self.x_prob = np.unique(self.x_label, return_counts=True)[1]
                self.x_prob = (self.x_prob / self._n).reshape(1, -1)
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
        """Estimate the approximation bound of the causal effect of the treatment
        on the outcome.

        Parameters
        ----------
        data : pandas.DataFrame, optional. Default to None
            Test data. The model will use the training data if set as None.
        
        treat : ndarray of str, optional. Defaults to None
            Values of the treatment group. For example, when there are multiple
            discrete treatments, array(['run', 'read']) means the treat value of
            the first treatment is taken as 'run' and that of the second treatment
            is taken as 'read'.
        
        control : ndarray of str, optional. Defaults to None
            Values of the control group.
        
        y_upper : float. Defaults to None
            The upper bound of the outcome.
        
        y_lower : float. Defaults to None
            The lower bound of the outcome.
        
        assump : str, optional.  Defaults to 'no-assump'.
            Options for the returned bounds. Should be one of
                1. no-assump: calculate the no assumption bound whose result
                    will always contain 0.
                2. non-negative: The treatment is always positive.
                3. non-positive: The treatment is always negative.
                4. optimal: The treatment is taken if its effect is positive.


        Returns
        -------
        tuple
            The first element is the lower bound while the second element is the
            upper bound. Note that if covariate is provided, all elements are 
            ndarrays of shapes (n, ) indicating the lower and upper bounds of 
            corresponding examples where n is the number of examples. 

        Raises
        ------
        Exception
            Raise Exception if the model is not fitted.
        
        Exception
            Raise Exception if the assump is not given correctly.
        """
        if not self._is_fitted:
            raise Exception(
                'The estimator is not fitted yet. Pleas call the fit method first.'
            )

        normal, optim1, optim2 = self._prepare4est(
            data, treat, control, y_upper, y_lower
        )
        upper, lower = normal
        zero_ = np.zeros_like(lower)
        
        if assump == 'no-assump' or assump is None:
            return (lower, upper)
        elif assump == 'non-negative':
            return (zero_, upper)
        elif assump == 'non-positive':
            return (lower, zero_)
        elif assump == 'optimal':
            # TODO: need to update this for discrete treatment
            return (optim1, optim2)
        else:
            raise Exception(
                'Only support assumptions in no-assump, non-negative, and'
                f'non-positive. But was given {assump}'
            )

    def comp_transormer(self, x, categories='auto'):
        """Transform the discrete treatment into one-hot vectors properly.

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

        if not self._is_fitted:
            self.oh_transformer = OneHotEncoder(categories=categories)
            self.oh_transformer.fit(x)

        x = self.oh_transformer.transform(x).toarray()

        return x

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

        treat = get_tr_ctrl(
            treat,
            self.comp_transormer,
            treat=True,
            one_hot=False,
            discrete_treat=True
        )
        control = get_tr_ctrl(
            control,
            self.comp_transormer,
            treat=False,
        )

        if self.covariate is None:
            n = 1
            v = None
            if data is None:
                y = self._y
                x = self.x_label
                # y_upper = y.max(axis=0) if y_upper is None else y_upper
                # y_lower = y.min(axis=0) if y_lower is None else y_lower
            else:
                y, x = convert2array(data, self.outcome, self.treatment)
                x = convert4onehot(self.comp_transormer(x,))
        else:
            if data is None:
                v = self._v
                y = self._y
                x = self.x_label
                n = self._n
            else:
                y, v, x = convert2array(
                    data, self.outcome, self.covariate, self.treatment
                )
                n = y.shape[0]
                x_prob = self.x_model.predict_proba(v)

        y_treat = get_groups(treat, x.reshape(-1, 1), False, y)[0]
        y_ctrl = get_groups(control, x.reshape(-1, 1), False, y)[0]
        y_tr_max, y_tr_min = y_treat.max(axis=0), y_treat.min(axis=0)
        y_ctrl_max, y_ctrl_min = y_ctrl.max(axis=0), y_ctrl.min(axis=0)

        y_upper = max(y_tr_max, y_ctrl_max) if y_upper is None else y_upper
        y_lower = min(y_tr_min, y_ctrl_min) if y_lower is None else y_lower

        # TODO: modify the following line for multiple treatment
        xt, x0 = np.zeros((n, self._x_d)), np.zeros((n, self._x_d))
        xt[:, treat] = 1
        x0[:, control] = 1

        xt = get_wv(xt, v)
        x0 = get_wv(x0, v)
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
