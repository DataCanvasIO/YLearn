import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from ylearn.estimator_model.iv import NP2SLS
from . import _dgp


def validate_it(data_generator, estimator,
                fit_kwargs=None, estimate_kwargs=None,
                check_fitted=True, check_effect=True):
    # generate data
    data, test_data, outcome, treatment, adjustment, covariate = data_generator()
    assert adjustment is not None

    instrument = adjustment[:3]
    adjustment = adjustment[3:]

    # fit
    kwargs = dict(adjustment=adjustment, instrument=instrument)
    if covariate:
        kwargs['covariate'] = covariate

    if fit_kwargs:
        kwargs.update(fit_kwargs)
    estimator.fit(data, outcome, treatment, **kwargs)
    if check_fitted:
        assert hasattr(estimator, '_is_fitted') and getattr(estimator, '_is_fitted')

    # estimate
    kwargs = dict(data=test_data, quantity=None)
    if estimate_kwargs:
        kwargs.update(estimate_kwargs)
    pred = estimator.estimate(**kwargs)
    assert pred is not None
    if check_effect:
        assert pred.min() != pred.max()

    # return leaner, pred


_test_settings = {
    # data_generator: (y_model,x_model,x_prob)
    _dgp.generate_data_x1b_y1: (LinearRegression(), RandomForestClassifier()),
    _dgp.generate_data_x2b_y1: (LinearRegression(), RandomForestClassifier()),
    _dgp.generate_data_x1m_y1: (LinearRegression(), RandomForestClassifier()),
}


@pytest.mark.parametrize('dg', _test_settings.keys())
# @pytest.mark.xfail(reason='to be fixed: effect is tuple')
def test_iv_with_params(dg):
    y_model, x_model = _test_settings[dg]
    dr = NP2SLS(x_model=x_model, y_model=y_model,
                is_discrete_treatment=True,
                is_discrete_outcome=False)
    validate_it(dg, dr)


def test_iv_basis():
    n = 5000

    # Initialize exogenous variables; normal errors, uniformly distributed covariates and instruments
    e = np.random.normal(size=(n,)) / 5
    x = np.random.uniform(low=0.0, high=10.0, size=(n,))
    z = np.random.uniform(low=0.0, high=10.0, size=(n,))

    # Initialize treatment variable
    # t = np.sqrt((x + 2) * z) + e
    t = np.sqrt(2 * z + x * z + x * x + x) + e
    # Show the marginal distribution of t
    y = t * t / 10 + x * x / 50 + e

    data_dict = {
        'z': z,
        'w': x,
        'x': t,
        'y': y
    }
    data = pd.DataFrame(data_dict)

    iv = NP2SLS()
    iv.fit(
        data=data,
        outcome='y',
        treatment='x',
        instrument='z',
        covariate='w',
        covar_basis=('Poly', 2),
        treatment_basis=('Poly', 2),
        instrument_basis=('Poly', 1),
    )
    for i, x in enumerate([2, 5, 8]):
        # y_true = t*t / 10 - x*t/10
        y_true = t * t / 10 + x * x / 50

        test_data = pd.DataFrame(
            {'x': t,
             'w': np.full_like(t, x), }
        )
        y_pred = iv.estimate(data=test_data)
        s = r2_score(y_true, y_pred)
        print('score:', s, f'x={x}')
