from ast import Or
from msilib.schema import SelfReg
from re import X
from xml.etree.ElementTree import tostring
import numpy as np

from sklearn import clone
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from estimator_model.base_models import BaseEstLearner
from estimator_model.utils import (convert2array, nd_kron)


class TwoSLS(BaseEstLearner):
    # TODO: simply import the 2SLS from StatsModel can finish this
    def __init__(
        self,
        random_state=2022,
        is_discrete_treatment=False,
        is_discrete_outcome=False,
        categories='auto'
    ):
        super().__init__(
            random_state,
            is_discrete_treatment,
            is_discrete_outcome,
            categories
        )

    def fit(self, data, outcome, treatment, **kwargs):
        return super().fit(data, outcome, treatment, **kwargs)

    def estimate(self, data=None, **kwargs):
        return super().estimate(data, **kwargs)

    def _prepare4est(self):
        pass


class NP2SLS(BaseEstLearner):
    r"""
    See Instrumental variable estimation of nonparametric models
    (https://eml.berkeley.edu/~powell/npiv.pdf) for reference.

    This method is similar to the conventional 2SLS and is also composed of
    2 stages after finding new features of x, w, and z,  
        f^d = f^d(z)
        g^\mu = g^\mu(v),
    which are represented by some non-linear functions (basis functions). These
    stages are:
        1. Fit the treatment model:
            x_hat(z, v, w) = a^{d, \mu} f_d(z)g_{\mu}(v) + h(v, w) + \eta
        2. Generate new treatments x_hat, and then fit the outcome model
            y(x_hat, v, w) = a^{m, \mu} \psi_m(x_hat)g_{\mu}(v) + k(v, w) 
            + \epsilon.
    The final causal effect is then estimated as
        y(x_hat_1, v, w) - y(x_hat_0, v, w).
    """

    def __init__(
        self,
        x_model=None,
        y_model=None,
        random_state=2022,
        is_discrete_treatment=False,
        is_discrete_outcome=False,
        categories='auto'
    ):
        self.x_model = LinearRegression() if x_model is None else x_model
        self.y_model = LinearRegression() if y_model is None else y_model

        super().__init__(
            random_state,
            is_discrete_treatment,
            is_discrete_outcome,
            categories
        )

    def fit(
        self,
        data,
        outcome,
        treatment,
        instrument,
        is_discrete_instrument=False,
        treatment_basis=None,
        instrument_basis=None,
        covar_basis=None,
        adjustment=None,
        covariate=None,
    ):
        assert instrument is not None, 'Instrument is required.'

        if all(
            treatment_basis is None, instrument_basis is None,
            covar_basis is None,
        ):
            raise ValueError(
                'Please specifiy the non-linear transformers for variables, '
                'or use TwoSLS method instead.'
            )

        super().fit(
            data, outcome, treatment,
            covariate=covariate,
            adjustment=adjustment,
            instrument=instrument,
            is_discrete_instrument=is_discrete_instrument
        )

        # data preprocessing
        y, x, z, w, v = convert2array(
            data, outcome, treatment, instrument, adjustment, covariate
        )

        if self.is_discrete_treatment:
            self.treatment_transformer = OrdinalEncoder()
            self.treatment_transformer.fit(x)
            x = self.treatment_transformer.transform(x)

        if self.is_discrete_instrument:
            self.instrument_transformer = OneHotEncoder()
            self.instrument_transformer.fit(z)
            z = self.instrument_transformer.transform(z)

        if isinstance(treatment_basis, list) \
           or isinstance(treatment_basis, tuple):
            method, degree = treatment_basis[0], treatment_basis[1]
            self._x_basis_func = self.get_basis_func(degree, method, x)
        else:
            self._x_basis_func = clone(treatment_basis)

        if isinstance(instrument_basis, list) \
           or isinstance(instrument_basis, tuple):
            method, degree = instrument_basis[0], instrument_basis[1]
            self._z_basis_func = self.get_basis_func(degree, method, z)
        else:
            self._z_basis_func = clone(instrument_basis)

        if isinstance(covar_basis, str) or isinstance(covar_basis, tuple):
            method, degree = covar_basis[0], covar_basis[1]
            self._v_basis_func = self.get_basis_func(degree, method, v)
        else:
            self._v_basis_func = clone(covar_basis)

        # get expansion basis functions for parameters
        x_transformed = self._x_basis_func(x)
        z_transformed = self._z_basis_func(x)
        v_tranformerd = self._v_basis_func(x)
        
        self._v_transformed = v_tranformerd
        self._w = w

        zvw = np.concatenate(
            [nd_kron(z_transformed, v_tranformerd), w, np.ones(x.shape[0], 1)],
            axis=1
        )

        # stage 1: fit the x_model on z and w, v
        self.x_model.fit(zvw, x_transformed)

        # stage 2: fit the y_model on x, and w, v
        x_hat = self.x_model.predict(zvw)
        xvw = np.concatenate(
            [nd_kron(x_hat, v_tranformerd), w, np.ones(x.shape[0], 1)], axis=1
        )
        self.y_model.fit(xvw, y)

        # set _is_fitted as True
        self._is_fitted = True

    def estimate(
        self,
        data=None,
        treat=None,
        control=None,
        quantity=None,
        marginal_effect=False,
    ):
        if not self._is_fitted:
            raise Exception('The estimator is not fitted yet.')

        yt, y0 = self._prepare4est(
            data=data,
            treat=treat,
            control=control,
            marginal_effect=marginal_effect,
        )
        if quantity == 'CATE':
            assert self.covariate is not None,\
                'Need caovariate to compute the CATE in this case.'
            return (yt - y0).mean(dim=0)
        if quantity == 'ATE':
            return (yt - y0).mean(dim=0)
        else:
            return (yt - y0)

    def _prepare4est(
        self,
        data,
        treat,
        control,
    ):
        if data is None:
            v = self._v_transformed
            w = self._w
        else:
            w, v = convert2array(
                data, self.adjustment, self.covariate
            )
            v = self._v_basis_func(v)

        n = w.shape[0]

        treat = 1 if treat is None \
            else self.treatment_transformer.transform(treat)
        control = 0 if control is None else \
            self.treatment_transformer.transform(control)

        xt = np.repeat(np.array([[treat]]), n, axis=0)
        x0 = np.repeat(np.array([[control]]), n, axis=0)
        xt = self._x_basis_func(xt)
        x0 = self._x_basis_func(x0)

        xvw_t = np.concatenate([nd_kron(xt, v), w, np.ones((n, 1))], aixs=1)
        xvw_0 = np.concatenate([nd_kron(x0, v), w, np.ones((n, 1))], aixs=1)

        yt = self.y_model.predict(xvw_t)
        y0 = self.y_model.predict(xvw_0)

        return yt, y0

    def get_basis_func(self, degree=5, func='Poly', *para):
        if func == 'Poly':
            return self._poly_basis_func(degree, *para)
        elif func == 'Hermite':
            return self._hermite_basis_func(degree, *para)
        else:
            raise ValueError('Do not support other basis functions currently.')

    def _poly_basis_func(self, degree, *para):
        pass

    def _hermite_basis_func(self, degree, *para):
        pass
