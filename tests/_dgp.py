import numpy as np
from numpy.random import binomial, multivariate_normal, uniform
from sklearn.model_selection import train_test_split

from ylearn.utils import to_df

TRAIN_SIZE = 1000
TEST_SIZE = 200
ADJUSTMENT_COUNT = 5
COVARIATE_COUNT = 3


def generate_variates(n, d):
    return multivariate_normal(np.zeros(d), np.diag(np.ones(d)), n)


def filter_columns(df, prefix):
    return list(filter(lambda c: c.startswith(prefix), df.columns.tolist()))


def generate_data(n_train, n_test, d_adjustment, d_covariate, fn_treatment, fn_outcome):
    """Generates population data for given untreated_outcome, treatment_effect and propensity functions.

    Parameters
    ----------
    n_train (int): train data size
    n_test (int): test data size
    d_adjustment (int): number of adjustments
    d_covariate (int): number of covariates
    fn_treatment (func<w,x>): untreated outcome conditional on covariates
    fn_outcome (func<w>): treatment effect conditional on covariates
    """

    # Generate covariates
    # W = multivariate_normal(np.zeros(d), np.diag(np.ones(d)), n)
    assert d_adjustment is not None or d_covariate is not None
    W = generate_variates(n_train + n_test, d_adjustment) if d_adjustment else None
    V = generate_variates(n_train + n_test, d_covariate) if d_covariate else None

    # Generate treatment
    fn_x = np.vectorize(fn_treatment, signature='(n)->(m)')
    X = fn_x(W) if W is not None else fn_x(V)

    # Calculate outcome
    fn_y = np.vectorize(fn_outcome, signature='(n),(m)->(k)')
    Y = fn_y(W, X) if W is not None else fn_y(V, X)

    # x
    data = to_df(w=W, x=X, y=Y, v=V)
    outcome = filter_columns(data, 'y')
    treatment = filter_columns(data, 'x')
    adjustment = filter_columns(data, 'w')
    covariate = filter_columns(data, 'v')
    if len(covariate) == 0:
        covariate = None

    if n_test is not None:
        # W_test = generate_variates(n_test, d_adjustment)
        # V_test = generate_variates(n_test, d_covariate) if d_covariate else None
        # if cut_test_at is not None:
        #     delta = 6 / n_test
        #     W_test[:, cut_test_at] = np.arange(-3, 3, delta)
        # test_data = to_df(w=W_test, v=V_test) if n_test is not None else None
        data, test_data = train_test_split(data, test_size=n_test)
    else:
        test_data = None

    return data, test_data, outcome, treatment, adjustment, covariate


def binary_TE(w):
    return 8 if w[1] > 0.1 else 0


def multiclass_TE(w, wi=1):
    boundary = [-1., -0.15, 0.15, 1.]
    return np.searchsorted(boundary, w[wi])


def generate_data_x1b_y1(train_size=TRAIN_SIZE, test_size=TEST_SIZE,
                         d_adjustment=ADJUSTMENT_COUNT, d_covariate=COVARIATE_COUNT):
    beta = uniform(-3, 3, d_adjustment if d_adjustment else d_covariate)

    def to_treatment(w):
        propensity = 0.8 if -0.5 < w[2] < 0.5 else 0.2
        return np.random.binomial(1, propensity, 1)

    def to_outcome(w, x):
        treatment_effect = binary_TE(w)
        y0 = np.dot(w, beta) + np.random.normal(0, 1)
        y = y0 + treatment_effect * x
        return y

    return generate_data(train_size, test_size, d_adjustment, d_covariate,
                         fn_treatment=to_treatment, fn_outcome=to_outcome)


def generate_data_x1b_y1_w5v0():
    return generate_data_x1b_y1(d_adjustment=5, d_covariate=0)


def generate_data_x1b_y1_w0v5():
    return generate_data_x1b_y1(d_adjustment=0, d_covariate=5)


def generate_data_x1b_y2(train_size=TRAIN_SIZE, test_size=TEST_SIZE,
                         d_adjustment=ADJUSTMENT_COUNT, d_covariate=COVARIATE_COUNT):
    beta = uniform(-3, 3, d_adjustment if d_adjustment else d_covariate)

    def to_treatment(w):
        propensity = 0.8 if -0.5 < w[2] < 0.5 else 0.2
        return np.random.binomial(1, propensity, 1)

    def to_outcome(w, x):
        treatment_effect = binary_TE(w)
        y0 = np.dot(w, beta) + np.random.normal(0, 1)
        y = y0 + treatment_effect * x + w[:2]
        return y

    return generate_data(train_size, test_size, d_adjustment, d_covariate,
                         fn_treatment=to_treatment, fn_outcome=to_outcome, )


def generate_data_x1b_y2_w5v0():
    return generate_data_x1b_y2(d_adjustment=5, d_covariate=0)


def generate_data_x1b_y2_w0v5():
    return generate_data_x1b_y2(d_adjustment=0, d_covariate=5)


def generate_data_x2b_y1(train_size=TRAIN_SIZE, test_size=TEST_SIZE,
                         d_adjustment=ADJUSTMENT_COUNT, d_covariate=COVARIATE_COUNT):
    beta = uniform(-3, 3, d_adjustment if d_adjustment else d_covariate)

    def to_treatment(w):
        propensity = 0.8 if -0.5 < w[2] < 0.5 else 0.2
        return np.random.binomial(1, propensity, 2)

    def to_outcome(w, x):
        treatment_effect = 8 if w[1] > 0.1 else 0
        y0 = np.dot(w, beta) + np.random.normal(0, 1)
        y = y0 + treatment_effect * x.mean()
        return np.array([y])

    return generate_data(train_size, test_size, d_adjustment, d_covariate,
                         fn_treatment=to_treatment, fn_outcome=to_outcome)


def generate_data_x2b_y1_w5v0():
    return generate_data_x2b_y1(d_adjustment=5, d_covariate=0)


def generate_data_x2b_y1_w0v5():
    return generate_data_x2b_y1(d_adjustment=0, d_covariate=5)


def generate_data_x2b_y2(train_size=TRAIN_SIZE, test_size=TEST_SIZE,
                         d_adjustment=ADJUSTMENT_COUNT, d_covariate=COVARIATE_COUNT):
    beta = uniform(-3, 3, d_adjustment if d_adjustment else d_covariate)

    def to_treatment(w):
        propensity = 0.8 if -0.5 < w[2] < 0.5 else 0.2
        return np.random.binomial(1, propensity, 2)

    def to_outcome(w, x):
        treatment_effect = np.array([8 if w[0] > 0.0 else 0,
                                     8 if w[1] > 0.1 else 0, ])
        y0 = np.dot(w, beta) + np.random.normal(0, 1)
        y = y0 + treatment_effect * x
        return y

    return generate_data(train_size, test_size, d_adjustment, d_covariate,
                         fn_treatment=to_treatment, fn_outcome=to_outcome)


def generate_data_x2b_y2_w5v0():
    return generate_data_x2b_y2(d_adjustment=5, d_covariate=0)


def generate_data_x2b_y2_w0v5():
    return generate_data_x2b_y2(d_adjustment=0, d_covariate=5)


def generate_data_x1m_y1(train_size=TRAIN_SIZE, test_size=TEST_SIZE,
                         d_adjustment=ADJUSTMENT_COUNT, d_covariate=COVARIATE_COUNT):
    beta = uniform(-3, 3, d_adjustment if d_adjustment else d_covariate)

    def to_treatment(w):
        # propensity = 0.8 if -0.5 < w[2] < 0.5 else 0.2
        # return np.random.binomial(1, propensity, 1)
        return np.array([multiclass_TE(w, wi=2), ])

    def to_outcome(w, x):
        treatment_effect = multiclass_TE(w)
        y0 = np.dot(w, beta) + np.random.normal(0, 1)
        y = y0 + treatment_effect * x
        return y

    return generate_data(train_size, test_size, d_adjustment, d_covariate,
                         fn_treatment=to_treatment, fn_outcome=to_outcome)


def generate_data_x1m_y1_w5v0():
    return generate_data_x1m_y1(d_adjustment=5, d_covariate=0)


def generate_data_x1m_y1_w0v5():
    return generate_data_x1m_y1(d_adjustment=0, d_covariate=5)


def generate_data_x2mb_y1(train_size=TRAIN_SIZE, test_size=TEST_SIZE,
                          d_adjustment=ADJUSTMENT_COUNT, d_covariate=COVARIATE_COUNT):
    beta = uniform(-3, 3, d_adjustment if d_adjustment else d_covariate)

    def to_treatment(w):
        propensity = 0.8 if -0.5 < w[3] < 0.5 else 0.2
        return np.array([multiclass_TE(w, wi=2),
                         np.random.binomial(1, propensity, 1)[0],
                         ])

    def to_outcome(w, x):
        treatment_effect = multiclass_TE(w)
        y0 = np.dot(w, beta) + np.random.normal(0, 1)
        y = y0 + treatment_effect * x[:1]
        return y

    return generate_data(train_size, test_size, d_adjustment, d_covariate,
                         fn_treatment=to_treatment, fn_outcome=to_outcome)


def generate_data_x2mb_y1_w5v0():
    return generate_data_x2mb_y1(d_adjustment=5, d_covariate=0)


def generate_data_x2mb_y1_w0v5():
    return generate_data_x2mb_y1(d_adjustment=0, d_covariate=5)
