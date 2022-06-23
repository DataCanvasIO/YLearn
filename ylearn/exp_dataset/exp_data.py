from random import random
import numpy as np
import scipy.special
from numpy.random import binomial, multivariate_normal, normal, uniform
import pandas as pd

from itertools import product

from sklearn.model_selection import train_test_split


def eta_sample(n):
    return np.random.uniform(-1, 1, size=n)


def epsilon_sample(n):
    return np.random.uniform(-1, 1, size=n)


def exp_te(x):
    return np.exp(2*x[0])


def ln_te(x):
    return np.log(1+x[0])


def build_data_frame(
    confounder_n,
    covariate_n,
    w, v, y, x, **kwargs,
):
    data_dict = {}
    for i in range(confounder_n):
        data_dict[f'w_{i}'] = w[:, i]

    for i in range(covariate_n):
        data_dict[f'c_{i}'] = v[:, i]

    def multi_var(x, name):
        if len(x.shape) == 1 or x.shape[1] == 1:
            data_dict[name] = x
        else:
            for i in range(x.shape[1]):
                data_dict[f'{name}_{i}'] = x[:, i]

    multi_var(x, 'treatment')
    multi_var(y, 'outcome')

    for k, v in kwargs.items():
        data_dict[k] = v

    data = pd.DataFrame(data_dict)
    train, val = train_test_split(data)
    
    return train, val


def coupon_dataset(n_users, treatment_style='binary', with_income=False):
    if with_income:
        income = np.random.normal(500, scale=15, size=n_users)
        gender = np.random.randint(0, 2, size=n_users)
        coupon = gender * 20 + 110 + income / 50 \
            + np.random.normal(scale=5, size=n_users)
        if treatment_style == 'binary':
            coupon = (coupon > 120).astype(int)
        amount = coupon * 150 + gender * 100 + 150 \
            + income / 5 + np.random.normal(size=n_users)
        time_spent = coupon * 10 + amount / 10

        df = pd.DataFrame({
            'gender': gender,
            'coupon': coupon,
            'amount': amount,
            'income': income,
            'time_spent': time_spent,
        })
    else:
        gender = np.random.randint(0, 2, size=n_users)
        coupon = gender * 20 + 150 + np.random.normal(scale=5, size=n_users)
        if treatment_style == 'binary':
            coupon = (coupon > 150).astype(int)
        amount = coupon * 30 + gender * 100 \
            + 150 + np.random.normal(size=n_users)
        time_spent = coupon * 100 + amount / 10

        df = pd.DataFrame({
            'gender': gender,
            'coupon': coupon,
            'amount': amount,
            'time_spent': time_spent,
        })

    return df

def generate_controls_outcome(d):
    beta = uniform(-3, 3, d)
    return lambda x: np.dot(x, beta) + normal(0, 1)

def binary_data(n=2000, d=5, n_test=250):
    def _binary_data(n, d, ctrl, ce, propensity):
        np.random.seed(2022)
        # Generate covariates
        v = multivariate_normal(np.zeros(d), np.diag(np.ones(d)), n)
        # Generate treatment
        x = np.apply_along_axis(lambda x: binomial(1, propensity(x), 1)[0], 1, v)
        # Calculate outcome
        y0 = np.apply_along_axis(lambda x: ctrl(x), 1, v)
        treat_effect = np.apply_along_axis(lambda x: ce(x), 1, v)
        y = y0 + treat_effect * x
        return (y, x, v)

    treatment_effect = lambda x: (1 if x[1] > 0.1 else 0)*8
    propensity = lambda x: (0.8 if (x[2]>-0.5 and x[2]<0.5) else 0.2)
    controls_outcome = generate_controls_outcome(d)
    
    return _binary_data(n, d, controls_outcome, treatment_effect, propensity)

def sq_data(n, d, n_x, random_seed=123):
    np.random.seed(random_seed)
    true_te = lambda X: np.hstack([X[:, [0]]**2 + 1, np.ones((X.shape[0], n_x - 1))])
    v = np.random.normal(0, 1, size=(n, d))
    x = np.random.normal(0, 1, size=(n, n_x))
    for t in range(n_x):
        x[:, t] = np.random.binomial(1, scipy.special.expit(v[:, 0]))
    y = np.sum(true_te(v) * x, axis=1, keepdims=True) + np.random.normal(0, .5, size=(n, 1))
    return y, x, v

def multi_continuous_treatment(
    n=6000,
    n_w=30,
    n_v=5,
    random_seed=2022,
):
    np.random.seed(random_seed)

    support_size = 5
    support_Y = np.random.choice(
        np.arange(n_w), size=support_size, replace=False
    )

    coefs_Y = np.random.uniform(0, 1, size=support_size)
    support_T = support_Y
    coefs_T = np.random.uniform(0, 1, size=support_size)

    W = np.random.normal(0, 1, size=(n, n_w))
    X = np.random.uniform(0, 1, size=(n, n_v))

    TE1 = np.array([exp_te(x_i) for x_i in X])
    TE2 = np.array([ln_te(x_i) for x_i in X]).flatten()
    T = np.dot(W[:, support_T], coefs_T) + eta_sample(n)
    Y = TE1 * T + TE2 * T**2 + \
        np.dot(W[:, support_Y], coefs_Y) + epsilon_sample(n)
    T = T.reshape(-1, 1)
    x = np.concatenate((T, T**2), axis=1)

    train, val = build_data_frame(confounder_n=n_w,
                                  covariate_n=n_v,
                                  w=W, v=X, y=Y, x=x, te1=TE1, te2=TE2)
    # test data
    X_test = np.random.uniform(0, 1, size=(100, n_v))
    X_test[:, 0] = np.linspace(0, 1, 100)

    data_test_dic = {}
    for i in range(n_v):
        data_test_dic[f'c_{i}'] = X_test[:, i]
    data_test = pd.DataFrame(data_test_dic)

    expected_te1 = np.array([exp_te(x_i) for x_i in X_test])
    expected_te2 = np.array([ln_te(x_i) for x_i in X_test]).flatten()

    return train, val, (data_test, expected_te1, expected_te2)


def single_binary_treatment(
    n=1000,
    confounder_n=30,
    covariate_n=4,
    random_seed=2022,
):
    np.random.seed(random_seed)
    support_size = 5
    # Outcome support
    support_Y = np.random.choice(
        range(confounder_n), size=support_size, replace=False)
    coefs_Y = np.random.uniform(0, 1, size=support_size)
    # Treatment support
    support_T = support_Y
    coefs_T = np.random.uniform(0, 1, size=support_size)

    # Generate controls, covariates, treatments and outcomes
    W = np.random.normal(0, 1, size=(n, confounder_n))
    V = np.random.uniform(0, 1, size=(n, covariate_n))
    # Heterogeneous treatment effects
    TE = np.array([exp_te(x_i) for x_i in V])
    # Define treatment
    log_odds = np.dot(W[:, support_T], coefs_T) + eta_sample(n)
    T_sigmoid = 1/(1 + np.exp(-log_odds))
    x = np.array([np.random.binomial(1, p) for p in T_sigmoid])
    # Define the outcome
    Y = TE * x + np.dot(W[:, support_Y], coefs_Y) + epsilon_sample(n)
    train, val = build_data_frame(confounder_n=confounder_n,
                                  covariate_n=covariate_n,
                                  w=W, v=V, y=Y, x=x, TE=TE)

    return train, val, TE


def single_continuous_treatment(num=2000,
                                confounder_n=30,
                                covariate_n=1,
                                random_seed=2022,
                                data_frame=True):
    np.random.seed(random_seed)
    support_size = 5
    support_y = np.random.choice(
        np.arange(confounder_n), size=support_size, replace=False
    )
    coefs_y = np.random.uniform(0, 1, size=support_size)
    support_t = support_y
    coefs_t = np.random.uniform(0, 1, size=support_size)
    w = np.random.normal(0, 1, size=(num, confounder_n))
    c = np.random.uniform(0, 1, size=(num, covariate_n))
    TE = np.array([exp_te(ci) for ci in c])
    x = np.dot(w[:, support_t], coefs_t) + eta_sample(num)
    y = TE * x + np.dot(w[:, support_y], coefs_y)\
        + epsilon_sample(num)

    x_test = np.array(list(product(np.arange(0, 1, 0.01), repeat=covariate_n)))
    if data_frame:
        train, val = build_data_frame(confounder_n=confounder_n,
                                      covariate_n=covariate_n,
                                      w=w, v=c, y=y, x=x, TE=TE)
        return train, val, TE


def meaningless_discrete_dataset_(num, treatment_effct,
                                  confounder_n=2,
                                  w_var=5,
                                  eps=1e-4,
                                  data_frame=True,
                                  random_seed=2022,
                                  instrument=False):
    """Generate a dataset where the treatment and outcome have some
    confounders while the relation between the treatment and outcome
    is linear. The treatment is an array of integers where each integer
    indicates the treatment group assigned to the corresponding example.
    The outcome is an array of float, i.e., we are building continuous
    outcome.

    Parameters
    ----------
    num : int
        The number of examples in the dataset.
    confounder_n : int
        The number of confounders of the treatment and outcome.
    treatment_effct : list, optional. Defaults to None.
    w_var : float, optional. Defaults to 0.5.
        Variance of the confounder around its mean.
    eps : float, optional. Defaults to 1e-4.
        Noise level imposed to the data generating process.
    data_frame : bool, optional. Defaults to True.
        Return pandas.DataFrame if True.
    random_seed : int, optional. Defaults to 2022.
    instrument : bool, optional. Defaults to False.
        Add instrument variables to the dataset if True.

    Returns
    ----------
    pandas.DataFrame, optional.
        w_j's are confounders of outcome and treatment.
    """
    np.random.seed(random_seed)

    # Build treatment x which depends on the confounder w
    x_num = len(treatment_effct)
    w = [
        np.random.normal(0, w_var*np.random.random_sample(), size=(num, 1))
        for i in range(confounder_n)
    ]
    w = np.concatenate(tuple(w), axis=1)
    w_coef = np.random.rand(x_num, confounder_n)
    x = w.dot(w_coef.T) + np.random.normal(0, eps, size=(num, x_num))
    if instrument:
        z = None
    x = x.argmax(axis=1)
    x_one_hot = np.eye(x_num)[x]

    # Now we build the outcome y which depends on both x and w
    x_coef = np.random.randn(1, confounder_n)
    x_coef = np.concatenate(
        (np.array(treatment_effct).reshape(1, -1), x_coef), axis=1
    )
    x_ = np.concatenate((x_one_hot, w), axis=1)
    y = x_.dot(x_coef.T) + np.random.normal(0, eps, size=(num, 1))

    # Return the dataset
    if data_frame:
        data_dict = {}
        data_dict['treatment'] = x
        if instrument:
            data_dict['instrument'] = z
        for i, j in enumerate(w.T):
            data_dict[f'w_{i}'] = j
        data_dict['outcome'] = y.reshape(num,)
        data = pd.DataFrame(data_dict)
        return data
    else:
        if instrument:
            return (x, w, z, y)
        else:
            return (x, w, y)



def meaningless_discrete_dataset(num, confounder_n,
                                 treatment_effct=None,
                                 prob=None,
                                 w_var=0.5,
                                 eps=1e-4,
                                 coef_range=5e4,
                                 data_frame=True,
                                 random_seed=2022):
    np.random.seed(random_seed)
    samples = np.random.multinomial(num, prob)
    # build treatment x with shape (num,), where the number of types
    # of treatments is len(prob) and each treatment i is assigned a
    # probability prob[i]
    x = []
    for i, sample in enumerate(samples):
        x += [i for j in range(sample)]
    np.random.shuffle(x)

    # construct the confounder w
    w = [
        np.random.normal(0, w_var, size=(num,)) for i in range(confounder_n)
    ]
    for i, w_ in enumerate(w, 1):
        x = x + w_
    x = np.round(x).astype(int)
    for i, j in enumerate(x):
        if j > len(prob) - 1:
            x[i] = len(prob) - 1
        elif j < 0:
            x[i] = 0

    # construct the outcome y
    coef = np.random.randint(int(coef_range*eps), size=(confounder_n,))
    y = np.random.normal(eps, size=(num,))
    for i in range(len(y)):
        y[i] = y[i] + treatment_effct[x[i]] * x[i]
    for i, j in zip(coef, w):
        y += i * j

    if data_frame:
        data_dict = {}
        data_dict['treatment'] = x
        for i, j in enumerate(w):
            data_dict[f'w_{i}'] = j
        data_dict['outcome'] = y
        data = pd.DataFrame(data_dict)
        return data, coef
    else:
        return (x, w, y, coef)
