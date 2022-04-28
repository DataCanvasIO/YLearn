import numpy as np
import pandas as pd
from numpy.random import binomial, multivariate_normal, uniform
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, MultiTaskLasso

from estimator_model.meta_learner import SLearner, TLearner, XLearner


# Define DGP
def generate_covariates(n, d):
    return multivariate_normal(np.zeros(d), np.diag(np.ones(d)), n)


def to_df(**data):
    dfs = []
    for k, v in data.items():
        if len(v.shape) == 1:
            dfs.append(pd.Series(v, name=k))
        elif v.shape[1] == 1:
            dfs.append(pd.DataFrame(v, columns=[k]))
        else:
            dfs.append(pd.DataFrame(v, columns=[f'{k}_{i}' for i in range(v.shape[1])]))
    df = pd.concat(dfs, axis=1)
    return df


def filter_columns(df, prefix):
    return list(filter(lambda c: c.startswith(prefix), df.columns.tolist()))


def generate_data(n_train, n_test, d, fn_treatment, fn_outcome):
    """Generates population data for given untreated_outcome, treatment_effect and propensity functions.

    Parameters
    ----------
        n_train (int): train data size
        n_test (int): test data size
        d (int): number of covariates
        fn_treatment (func<w,x>): untreated outcome conditional on covariates
        fn_outcome (func<w>): treatment effect conditional on covariates
    """

    # Generate covariates
    # W = multivariate_normal(np.zeros(d), np.diag(np.ones(d)), n)
    W = generate_covariates(n_train, d)

    # Generate treatment
    fn_x = np.vectorize(fn_treatment, signature='(n)->(m)')
    X = fn_x(W)

    # Calculate outcome
    fn_y = np.vectorize(fn_outcome, signature='(n),(m)->(k)')
    Y = fn_y(W, X)

    # x
    data = to_df(w=W, x=X, y=Y)
    outcome = filter_columns(data, 'y')
    treatment = filter_columns(data, 'x')
    adjustment = filter_columns(data, 'w')

    X_test = generate_covariates(n_test, d)
    # delta = 6 / n_test
    # X_test[:, 1] = np.arange(-3, 3, delta)
    test_data = to_df(w=generate_covariates(n_test, d)) if n_test is not None else None

    return data, test_data, outcome, treatment, adjustment


def TE(w):
    return 8 if w[1] > 0.1 else 0


def generate_data_x1y1():
    d = 5
    beta = uniform(-3, 3, d)

    def to_treatment(w):
        propensity = 0.8 if -0.5 < w[2] < 0.5 else 0.2
        return np.random.binomial(1, propensity, 1)

    def to_outcome(w, x):
        treatment_effect = TE(w)
        y0 = np.dot(w, beta) + np.random.normal(0, 1)
        y = y0 + treatment_effect * x
        return y

    return generate_data(1000, 200, d, fn_treatment=to_treatment, fn_outcome=to_outcome)


def generate_data_x2y1():
    d = 5
    beta = uniform(-3, 3, d)

    def to_treatment(w):
        propensity = 0.8 if -0.5 < w[2] < 0.5 else 0.2
        return np.random.binomial(1, propensity, 2)

    def to_outcome(w, x):
        treatment_effect = 8 if w[1] > 0.1 else 0
        y0 = np.dot(w, beta) + np.random.normal(0, 1)
        y = y0 + treatment_effect * x.mean()
        return np.array([y])

    return generate_data(1000, 200, d, fn_treatment=to_treatment, fn_outcome=to_outcome)


def generate_data_x2y2():
    d = 5
    beta = uniform(-3, 3, d)

    def to_treatment(w):
        propensity = 0.8 if -0.5 < w[2] < 0.5 else 0.2
        return np.random.binomial(1, propensity, 2)

    def to_outcome(w, x):
        treatment_effect = np.array([8 if w[0] > 0.0 else 0,
                                     8 if w[1] > 0.1 else 0, ])
        y0 = np.dot(w, beta) + np.random.normal(0, 1)
        y = y0 + treatment_effect * x
        return y

    return generate_data(1000, 200, d, fn_treatment=to_treatment, fn_outcome=to_outcome)


def test_sleaner_x1y1():
    data, test_data, outcome, treatment, adjustment = generate_data_x1y1()

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


def test_sleaner_x1y1():
    data, test_data, outcome, treatment, adjustment = generate_data_x1y1()

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


def test_sleaner_x2y1():
    data, test_data, outcome, treatment, adjustment = generate_data_x2y1()

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


def test_sleaner_x2y2():
    data, test_data, outcome, treatment, adjustment = generate_data_x2y2()

    s = SLearner(model=LinearRegression())
    s.fit(
        data=data,
        outcome=outcome,
        treatment=treatment,
        adjustment=adjustment,
    )
    s_pred = s.estimate(data=test_data, quantity=None)

    s1 = SLearner(model=MultiTaskLasso())
    s1.fit(
        data=data,
        outcome=outcome,
        treatment=treatment,
        adjustment=adjustment,
        combined_treatment=False
    )
    s1_pred = s1.estimate(data=test_data, quantity=None)


def test_tleaner_x1y1():
    data, test_data, outcome, treatment, adjustment = generate_data_x1y1()

    t = TLearner(model=GradientBoostingRegressor())
    t.fit(
        data=data,
        outcome=outcome,
        treatment=treatment,
        adjustment=adjustment,
    )
    t_pred = t.estimate(data=test_data, quantity=None)

    t1 = TLearner(model=GradientBoostingRegressor())
    t1.fit(
        data=data,
        outcome=outcome,
        treatment=treatment,
        adjustment=adjustment,
        combined_treatment=False,
    )
    t1_pred = t1.estimate(data=test_data, quantity=None)


def test_tleaner_x2y1():
    data, test_data, outcome, treatment, adjustment = generate_data_x2y1()

    t = TLearner(model=GradientBoostingRegressor())
    t.fit(
        data=data,
        outcome=outcome,
        treatment=treatment,
        adjustment=adjustment,
    )
    t_pred = t.estimate(data=test_data, quantity=None)

    t1 = TLearner(model=GradientBoostingRegressor())
    t1.fit(
        data=data,
        outcome=outcome,
        treatment=treatment,
        adjustment=adjustment,
        combined_treatment=False,
    )
    t1_pred = t1.estimate(data=test_data, quantity=None)


def test_tleaner_x2y2():
    data, test_data, outcome, treatment, adjustment = generate_data_x2y2()

    t = TLearner(model=LinearRegression())
    t.fit(
        data=data,
        outcome=outcome,
        treatment=treatment,
        adjustment=adjustment,
    )
    t_pred = t.estimate(data=test_data, quantity=None)

    t1 = TLearner(model=LinearRegression())
    t1.fit(
        data=data,
        outcome=outcome,
        treatment=treatment,
        adjustment=adjustment,
        combined_treatment=False,
    )
    t1_pred = t1.estimate(data=test_data, quantity=None)


def test_xleaner_x1y1():
    data, test_data, outcome, treatment, adjustment = generate_data_x1y1()
    x = XLearner(model=GradientBoostingRegressor())
    x.fit(
        data=data,
        outcome=outcome,
        treatment=treatment,
        adjustment=adjustment,
    )
    x_pred = x.estimate(data=test_data, quantity=None)

    x1 = XLearner(model=GradientBoostingRegressor())
    x1.fit(
        data=data,
        outcome=outcome,
        treatment=treatment,
        adjustment=adjustment,
        combined_treatment=False,
    )
    x1_pred = x1.estimate(data=test_data, quantity=None)


def test_xleaner_x2y1():
    data, test_data, outcome, treatment, adjustment = generate_data_x2y1()
    x = XLearner(model=GradientBoostingRegressor())
    x.fit(
        data=data,
        outcome=outcome,
        treatment=treatment,
        adjustment=adjustment,
    )
    x_pred = x.estimate(data=test_data, quantity=None)

    x1 = XLearner(model=GradientBoostingRegressor())
    x1.fit(
        data=data,
        outcome=outcome,
        treatment=treatment,
        adjustment=adjustment,
        combined_treatment=False,
    )
    x1_pred = x1.estimate(data=test_data, quantity=None)


def test_xleaner_x2y2():
    data, test_data, outcome, treatment, adjustment = generate_data_x2y2()
    x = XLearner(model=LinearRegression())
    x.fit(
        data=data,
        outcome=outcome,
        treatment=treatment,
        adjustment=adjustment,
    )
    x_pred = x.estimate(data=test_data, quantity=None)

    x1 = XLearner(model=LinearRegression())
    x1.fit(
        data=data,
        outcome=outcome,
        treatment=treatment,
        adjustment=adjustment,
        combined_treatment=False,
    )
    x1_pred = x1.estimate(data=test_data, quantity=None)
