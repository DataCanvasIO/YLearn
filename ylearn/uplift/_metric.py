from collections import OrderedDict

import numpy as np
import pandas as pd


class Cumulator:
    def __init__(self, common_columns=None, random_name=None, random_column_number=10):
        assert common_columns is None or isinstance(common_columns, (list, tuple))

        self.common_columns = list(common_columns) if common_columns is not None else []
        self.random_name = random_name
        self.random_column_number = random_column_number

    def __call__(self, df, return_top_point=False):
        if self.random_name is None:
            result, top_point = self.cumulate(df)
        else:
            result, top_point = self.cumulate_with_random(df)

        if return_top_point:
            return result, top_point
        else:
            return result

    def cumulate(self, df):
        columns = df.columns.tolist()
        if self.common_columns:
            assert all(map(lambda c: c in columns, self.common_columns))
            columns = [c for c in columns if c not in self.common_columns]

        n = len(df)
        result = []
        result_top_point = OrderedDict()

        for col in columns:
            df_col = df[[col, ] + self.common_columns].sort_values(col, ascending=False)
            df_col.index = pd.RangeIndex(1, n + 1)
            r = self.cumulate_column(df_col, col)

            assert isinstance(r, (pd.DataFrame, pd.Series))
            result.append(r)
            idx = r.idxmax(axis=0, skipna=True)
            result_top_point[col] = df_col[col].loc[idx]

        result = pd.concat(result, join='inner', axis=1)
        result.loc[0] = np.zeros((result.shape[1],))
        result = result.sort_index().interpolate()

        return result, pd.Series(result_top_point)

    def cumulate_with_random(self, df):
        assert self.random_name is not None

        result, top_point = self.cumulate(df)
        result_random, _ = self.cumulate(self._generate_random_like(df))
        result[self.random_name] = result_random.mean(axis=1)

        return result, top_point

    def _generate_random_like(self, df_base):
        n_sample = len(df_base)
        data = {f'__{self.random_name}_{i}__': np.random.rand(n_sample) for i in range(self.random_column_number)}
        df = pd.DataFrame(data, index=df_base.index)
        df = pd.concat([df, df_base[self.common_columns]], axis=1)
        return df

    def cumulate_column(self, df_col, col_name):
        raise NotImplementedError()


class CumulatorWithTreatment(Cumulator):
    def __init__(self, outcome='y', treatment='x', treat=1, control=0,
                 random_name=None, random_column_number=10):
        self.treatment = treatment
        self.outcome = outcome
        self.treat = treat
        self.control = control

        super().__init__([outcome, treatment],
                         random_name=random_name,
                         random_column_number=random_column_number,
                         )


class CumulatorWithTrueEffect(Cumulator):
    def __init__(self, outcome='y', treatment='x', true_effect=None, treat=1, control=0,
                 random_name=None, random_column_number=10):
        assert true_effect is not None

        self.treatment = treatment
        self.outcome = outcome
        self.true_effect = true_effect
        self.treat = treat
        self.control = control

        super().__init__([c for c in [outcome, treatment, true_effect, ] if c is not None],
                         random_name=random_name,
                         random_column_number=random_column_number,
                         )


class LiftCumulatorWithTreatment(CumulatorWithTreatment):
    def cumulate_column(self, df_col, col_name):
        assert set(df_col[self.treatment].unique()) == {self.treat, self.control}

        idx_treat = df_col[self.treatment] == self.treat
        idx_control = df_col[self.treatment] == self.control

        df_col['__x_tr__'] = idx_treat.cumsum()
        df_col['__x_ct__'] = df_col.index.values - df_col['__x_tr__']
        df_col['__y_tr__'] = (df_col[self.outcome] * idx_treat).cumsum()
        df_col['__y_ct__'] = (df_col[self.outcome] * idx_control).cumsum()

        lift = df_col['__y_tr__'] / df_col['__x_tr__'] - df_col['__y_ct__'] / df_col['__x_ct__']
        lift.name = col_name
        return lift


class LiftCumulatorWithTrueEffect(CumulatorWithTrueEffect):
    def cumulate_column(self, df_col, col_name):
        lift = df_col[self.true_effect].cumsum() / df_col.index
        lift.name = col_name
        return lift


class GainCumulatorWithTreatment(LiftCumulatorWithTreatment):
    def cumulate_column(self, df_col, col_name):
        lift = super().cumulate_column(df_col, col_name)
        gain = lift.mul(lift.index.values, axis=0)
        gain.name = col_name
        return gain


class GainCumulatorWithTrueEffect(LiftCumulatorWithTrueEffect):
    def cumulate_column(self, df_col, col_name):
        lift = super().cumulate_column(df_col, col_name)
        gain = lift.mul(lift.index.values, axis=0)
        gain.name = col_name
        return gain


class QiniCumulatorWithTreatment(CumulatorWithTreatment):
    def cumulate_column(self, df_col, col_name):
        assert set(df_col[self.treatment].unique()) == {self.treat, self.control}

        idx_treat = df_col[self.treatment] == self.treat
        idx_control = df_col[self.treatment] == self.control

        df_col['__x_tr__'] = idx_treat.cumsum()
        df_col['__x_ct__'] = df_col.index.values - df_col['__x_tr__']
        df_col['__y_tr__'] = (df_col[self.outcome] * idx_treat).cumsum()
        df_col['__y_ct__'] = (df_col[self.outcome] * idx_control).cumsum()

        r = df_col['__y_tr__'] - df_col['__y_ct__'] * df_col['__x_tr__'] / df_col['__x_ct__']
        r.name = col_name
        return r


class QiniCumulatorWithTrueEffect(CumulatorWithTrueEffect):
    def cumulate_column(self, df_col, col_name):
        assert set(df_col[self.treatment].unique()) == {self.treat, self.control}

        idx_treat = df_col[self.treatment] == self.treat
        # idx_control = df_col[self.treatment] == self.control

        df_col['__x_tr__'] = idx_treat.cumsum()
        r = df_col[self.true_effect].cumsum() / df_col.index * df_col['__x_tr__']
        r.name = col_name
        return r


def get_cumlift(df, outcome='y', treatment='x', true_effect=None, treat=1, control=0, random_name='RANDOM'):
    if true_effect is not None:
        cumulator = LiftCumulatorWithTrueEffect(
            outcome=outcome, treatment=treatment, true_effect=true_effect,
            treat=treat, control=control, random_name=random_name)
    else:
        cumulator = LiftCumulatorWithTreatment(
            outcome=outcome, treatment=treatment,
            treat=treat, control=control, random_name=random_name)

    lift = cumulator(df, return_top_point=False)
    return lift


def get_gain(df, outcome='y', treatment='x', true_effect=None, treat=1, control=0,
             normalize=True, random_name='RANDOM', return_top_point=False):
    if true_effect is not None:
        cumulator = GainCumulatorWithTrueEffect(
            outcome=outcome, treatment=treatment, true_effect=true_effect,
            treat=treat, control=control, random_name=random_name)
    else:
        cumulator = GainCumulatorWithTreatment(
            outcome=outcome, treatment=treatment,
            treat=treat, control=control, random_name=random_name)

    gain, top_point = cumulator(df, return_top_point=True)
    if normalize:
        gain = gain.div(np.abs(gain.iloc[-1, :]), axis=1)

    if return_top_point:
        return gain, top_point
    else:
        return gain


def get_qini(df, outcome='y', treatment='x', true_effect=None, treat=1, control=0,
             normalize=True, random_name='RANDOM', return_top_point=False):
    if true_effect is not None:
        cumulator = QiniCumulatorWithTrueEffect(
            outcome=outcome, treatment=treatment, true_effect=true_effect,
            treat=treat, control=control, random_name=random_name)
    else:
        cumulator = QiniCumulatorWithTreatment(
            outcome=outcome, treatment=treatment,
            treat=treat, control=control, random_name=random_name)

    qini, top_point = cumulator(df, return_top_point=True)
    if normalize:
        qini = qini.div(np.abs(qini.iloc[-1, :]), axis=1)

    if return_top_point:
        return qini, top_point
    else:
        return qini


def gain_top_point(df, outcome='y', treatment='x', true_effect=None, treat=1, control=0):
    _, top_point = get_gain(df,
                            outcome=outcome, treatment=treatment, true_effect=true_effect,
                            treat=treat, control=control, return_top_point=True)
    return top_point


def qini_top_point(df, outcome='y', treatment='x', true_effect=None, treat=1, control=0):
    _, top_point = get_qini(df,
                            outcome=outcome, treatment=treatment, true_effect=true_effect,
                            treat=treat, control=control, return_top_point=True)
    return top_point


def auuc_score(df, outcome='y', treatment='x', true_effect=None, treat=1, control=0,
               normalize=True, random_name='RANDOM', ):
    gain = get_gain(df, outcome=outcome, treatment=treatment, true_effect=true_effect,
                    treat=treat, control=control,
                    normalize=normalize, random_name=random_name, )
    auuc = gain.sum() / len(gain)
    auuc.name = 'auuc'
    return auuc


def qini_score(df, outcome='y', treatment='x', true_effect=None, treat=1, control=0,
               normalize=True, random_name='RANDOM'):
    qini = get_qini(df, outcome=outcome, treatment=treatment, true_effect=true_effect,
                    treat=treat, control=control,
                    normalize=normalize,
                    random_name='RANDOM' if random_name is None else random_name)

    if random_name is None:
        qini = (qini.sum(axis=0) - qini['RANDOM'].sum()) / len(qini)
        qini = qini.drop('RANDOM')
    else:
        qini = (qini.sum(axis=0) - qini[random_name].sum()) / len(qini)
    qini.name = 'qini'
    return qini
