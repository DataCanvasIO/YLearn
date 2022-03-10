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
