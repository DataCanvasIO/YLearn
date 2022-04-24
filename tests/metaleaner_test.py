import numpy as np
from numpy.random import binomial, multivariate_normal, normal, uniform
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor

import pandas as pd

from estimator_model.meta_learner import SLearner, TLearner, XLearner
from estimator_model.doubly_robust import DoublyRobust


# Define DGP
def generate_data(n, d, controls_outcome, treatment_effect, propensity):
    """Generates population data for given untreated_outcome, treatment_effect and propensity functions.

    Parameters
    ----------
        n (int): population size
        d (int): number of covariates
        controls_outcome (func): untreated outcome conditional on covariates
        treatment_effect (func): treatment effect conditional on covariates
        propensity (func): probability of treatment conditional on covariates
    """
    # Generate covariates
    X = multivariate_normal(np.zeros(d), np.diag(np.ones(d)), n)
    # Generate treatment
    T = np.apply_along_axis(lambda x: binomial(1, propensity(x), 1)[0], 1, X)
    # Calculate outcome
    Y0 = np.apply_along_axis(lambda x: controls_outcome(x), 1, X)
    treat_effect = np.apply_along_axis(lambda x: treatment_effect(x), 1, X)
    Y = Y0 + treat_effect * T
    return (Y, T, X)


def generate_controls_outcome(d):
    beta = uniform(-3, 3, d)
    return lambda x: np.dot(x, beta) + normal(0, 1)


def generate_data_in1_out1():
    treatment_effect = lambda x: (1 if x[1] > 0.1 else 0) * 8
    propensity = lambda x: (0.8 if (x[2] > -0.5 and x[2] < 0.5) else 0.2)
    # DGP constants and test data
    d = 5
    n = 1000
    n_test = 250
    controls_outcome = generate_controls_outcome(d)
    X_test = multivariate_normal(np.zeros(d), np.diag(np.ones(d)), n_test)
    delta = 6 / n_test
    X_test[:, 1] = np.arange(-3, 3, delta)

    y, x, w = generate_data(n, d, controls_outcome, treatment_effect, propensity)
    data_dict = {
        'outcome': y,
        'treatment': x,
    }
    test_dict = {}
    adjustment = []
    for i in range(w.shape[1]):
        data_dict[f'w_{i}'] = w[:, i].squeeze()
        test_dict[f'w_{i}'] = X_test[:, i].squeeze()
        adjustment.append(f'w_{i}')
    outcome = 'outcome'
    treatment = 'treatment'
    data = pd.DataFrame(data_dict)
    test_data = pd.DataFrame(test_dict)

    return data, test_data, outcome, treatment, adjustment


def test_sleaner():
    data, test_data, outcome, treatment, adjustment = generate_data_in1_out1()

    s = SLearner(model=GradientBoostingRegressor())
    s.fit(
        data=data,
        outcome=outcome,
        treatment=treatment,
        adjustment=adjustment,
    )
    s_pred = s.estimate(data=test_data, quantity=None)

    s1 = SLearner(model=GradientBoostingRegressor())
    s1.fit(
        data=data,
        outcome=outcome,
        treatment=treatment,
        adjustment=adjustment,
        combined_treatment=False
    )
    s1_pred = s1.estimate(data=test_data, quantity=None)


def test_tleaner():
    data, test_data, outcome, treatment, adjustment = generate_data_in1_out1()

    t = TLearner(
        model=GradientBoostingRegressor()
    )
    t.fit(
        data=data,
        outcome=outcome,
        treatment=treatment,
        adjustment=adjustment,
    )
    t_pred = t.estimate(data=test_data, quantity=None)

    t1 = TLearner(
        model=GradientBoostingRegressor()
    )
    t1.fit(
        data=data,
        outcome=outcome,
        treatment=treatment,
        adjustment=adjustment,
        combined_treatment=False,
    )
    t1_pred = t1.estimate(data=test_data, quantity=None)


def test_xleaner():
    data, test_data, outcome, treatment, adjustment = generate_data_in1_out1()
    x = XLearner(
        model=GradientBoostingRegressor()
    )
    x.fit(
        data=data,
        outcome=outcome,
        treatment=treatment,
        adjustment=adjustment,
    )
    x_pred = x.estimate(data=test_data, quantity=None)

    x1 = XLearner(
        model=GradientBoostingRegressor()
    )
    x1.fit(
        data=data,
        outcome=outcome,
        treatment=treatment,
        adjustment=adjustment,
        combined_treatment=False,
    )
    x1_pred = x1.estimate(data=test_data, quantity=None)
