from typing import no_type_check
import numpy as np
import pandas as pd


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


def meaningless_discrete_dataset(num, confounder_num,
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
        np.random.normal(0, w_var, size=(num,)) for i in range(confounder_num)
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
    coef = np.random.randint(int(coef_range*eps), size=(confounder_num,))
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
        return x, w, y, coef
