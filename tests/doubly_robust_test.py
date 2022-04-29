import pytest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression

from ylearn.estimator_model.doubly_robust import DoublyRobust
from .metaleaner_test import generate_data_x1y1, generate_data_x1y2, generate_data_x2y1


def test_dr_x1y1():
    data, test_data, outcome, treatment, adjustment = generate_data_x1y1()
    n = len(data)

    dr = DoublyRobust(
        x_model=RandomForestClassifier(n_estimators=100, max_depth=100, min_samples_leaf=int(n / 100)),
        y_model=GradientBoostingRegressor(n_estimators=100, max_depth=100, min_samples_leaf=int(n / 100)),
        yx_model=GradientBoostingRegressor(n_estimators=100, max_depth=100, min_samples_leaf=int(n / 100)),
        cf_fold=1,
        random_state=2022,
    )
    dr.fit(
        data=data,
        outcome=outcome,
        treatment=treatment,
        covariate=adjustment,
    )
    dr_pred = dr.estimate(data=test_data, quantity=None)[1].squeeze()


def test_dr_x1y2():
    data, test_data, outcome, treatment, adjustment = generate_data_x1y2()
    n = len(data)

    dr = DoublyRobust(
        x_model=RandomForestClassifier(n_estimators=100, max_depth=100, min_samples_leaf=int(n / 100)),
        y_model=LinearRegression(),
        yx_model=LinearRegression(),
        cf_fold=1,
        random_state=2022,
    )
    dr.fit(
        data=data,
        outcome=outcome,
        treatment=treatment,
        covariate=adjustment,
    )
    dr_pred = dr.estimate(data=test_data, quantity=None)[1].squeeze()


@pytest.mark.xfail(reason='fix later')
def test_dr_x2y1():
    data, test_data, outcome, treatment, adjustment = generate_data_x2y1()
    n = len(data)

    dr = DoublyRobust(
        x_model=RandomForestClassifier(n_estimators=100, max_depth=100, min_samples_leaf=int(n / 100)),
        y_model=GradientBoostingRegressor(n_estimators=100, max_depth=100, min_samples_leaf=int(n / 100)),
        yx_model=GradientBoostingRegressor(n_estimators=100, max_depth=100, min_samples_leaf=int(n / 100)),
        cf_fold=1,
        random_state=2022,
    )
    dr.fit(
        data=data,
        outcome=outcome,
        treatment=treatment,
        covariate=adjustment,
    )
    dr_pred = dr.estimate(data=test_data, quantity=None)[1].squeeze()


@pytest.mark.xfail(reason='fix later')
def test_dr_x2y2():
    data, test_data, outcome, treatment, adjustment = generate_data_x2y1()
    n = len(data)

    dr = DoublyRobust(
        x_model=RandomForestClassifier(n_estimators=100, max_depth=100, min_samples_leaf=int(n / 100)),
        y_model=LinearRegression(),
        yx_model=LinearRegression(),
        cf_fold=1,
        random_state=2022,
    )
    dr.fit(
        data=data,
        outcome=outcome,
        treatment=treatment,
        covariate=adjustment,
    )
    dr_pred = dr.estimate(data=test_data, quantity=None)[1].squeeze()
