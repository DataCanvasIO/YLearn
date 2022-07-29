import numpy as np
import pandas as pd


class Cumulator:
    def __init__(self, common_columns=None, random_name=None, random_column_number=10):
        assert common_columns is None or isinstance(common_columns, (list, tuple))

        self.common_columns = list(common_columns) if common_columns is not None else []
        self.random_name = random_name
        self.random_column_number = random_column_number

    def __call__(self, df):
        if self.random_name is None:
            return self.cumulate(df)
        else:
            return self.cumulate_with_random(df)

    def cumulate(self, df):
        columns = df.columns.tolist()
        if self.common_columns:
            assert all(map(lambda c: c in columns, self.common_columns))
            columns = [c for c in columns if c not in self.common_columns]

        n = len(df)
        result = []

        for col in columns:
            df_col = df[[col, ] + self.common_columns].sort_values(col, ascending=False)
            df_col.index = pd.RangeIndex(1, n + 1)
            r = self.cumulate_column(df_col, col)

            assert isinstance(r, (pd.DataFrame, pd.Series))
            result.append(r)

        result = pd.concat(result, join='inner', axis=1)
        result.loc[0] = np.zeros((result.shape[1],))
        result = result.sort_index().interpolate()

        return result

    def cumulate_with_random(self, df):
        assert self.random_name is not None

        result = self.cumulate(df)
        result_random = self.cumulate(self._generate_random_like(df))
        result[self.random_name] = result_random.mean(axis=1)

        return result

    def _generate_random_like(self, df_base):
        n_sample = len(df_base)
        data = {f'__{self.random_name}_{i}__': np.random.rand(n_sample) for i in range(self.random_column_number)}
        df = pd.DataFrame(data, index=df_base.index)
        df = pd.concat([df, df_base[self.common_columns]], axis=1)
        return df

    def cumulate_column(self, df_col, col_name):
        raise NotImplementedError()


class CumulatorWithTreatment(Cumulator):
    def __init__(self, outcome='y', treatment='x', random_name=None, random_column_number=10):
        self.treatment = treatment
        self.outcome = outcome

        super().__init__([outcome, treatment],
                         random_name=random_name,
                         random_column_number=random_column_number,
                         )


class CumulatorWithTrueEffect(Cumulator):
    def __init__(self, outcome='y', treatment='x', true_effect=None, random_name=None, random_column_number=10):
        assert true_effect is not None

        self.treatment = treatment
        self.outcome = outcome
        self.true_effect = true_effect

        super().__init__([c for c in [outcome, treatment, true_effect, ] if c is not None],
                         random_name=random_name,
                         random_column_number=random_column_number,
                         )


class LiftCumulatorWithTreatment(CumulatorWithTreatment):
    def cumulate_column(self, df_col, col_name):
        df_col['__x_tr__'] = df_col[self.treatment].cumsum()
        df_col['__x_ct__'] = df_col.index.values - df_col['__x_tr__']
        df_col['__y_tr__'] = (df_col[self.outcome] * df_col[self.treatment]).cumsum()
        df_col['__y_ct__'] = (df_col[self.outcome] * (1 - df_col[self.treatment])).cumsum()

        lift = df_col['__y_tr__'] / df_col['__x_tr__'] - df_col['__y_ct__'] / df_col['__x_ct__']
        lift.name = col_name
        return lift


class LiftCumulatorWithTrueEffect(CumulatorWithTrueEffect):
    def cumulate_column(self, df_col, col_name):
        lift = df_col[self.true_effect].cumsum() / df_col.index
        lift.name = col_name
        return lift


class QiniCumulatorWithTreatment(CumulatorWithTreatment):
    def cumulate_column(self, df_col, col_name):
        df_col['__x_tr__'] = df_col[self.treatment].cumsum()
        df_col['__x_ct__'] = df_col.index.values - df_col['__x_tr__']
        df_col['__y_tr__'] = (df_col[self.outcome] * df_col[self.treatment]).cumsum()
        df_col['__y_ct__'] = (df_col[self.outcome] * (1 - df_col[self.treatment])).cumsum()

        r = df_col['__y_tr__'] - df_col['__y_ct__'] * df_col['__x_tr__'] / df_col['__x_ct__']
        r.name = col_name
        return r


class QiniCumulatorWithTrueEffect(CumulatorWithTrueEffect):
    def cumulate_column(self, df_col, col_name):
        df_col['__x_tr__'] = df_col[self.treatment].cumsum()
        r = df_col[self.true_effect].cumsum() / df_col.index * df_col['__x_tr__']
        r.name = col_name
        return r


def cum_lift(df, outcome='y', treatment='x', true_effect=None, random_name='RANDOM'):
    if true_effect is not None:
        cumulator = LiftCumulatorWithTrueEffect(outcome=outcome, treatment=treatment, true_effect=true_effect,
                                                random_name=random_name)
    else:
        cumulator = LiftCumulatorWithTreatment(outcome=outcome, treatment=treatment, random_name=random_name)

    lift = cumulator(df)
    return lift


def get_gain(df, outcome='y', treatment='x', true_effect=None, normalize=True, random_name='RANDOM'):
    lift = cum_lift(df, outcome=outcome, treatment=treatment,
                    true_effect=true_effect,
                    random_name=random_name,
                    )

    gain = lift.mul(lift.index.values, axis=0)
    if normalize:
        gain = gain.div(np.abs(gain.iloc[-1, :]), axis=1)
    return gain


def get_qini(df, outcome='y', treatment='x', true_effect=None, normalize=True, random_name='RANDOM'):
    if true_effect is not None:
        cumulator = QiniCumulatorWithTrueEffect(outcome=outcome, treatment=treatment, true_effect=true_effect,
                                                random_name=random_name)
    else:
        cumulator = QiniCumulatorWithTreatment(outcome=outcome, treatment=treatment, random_name=random_name)

    qini = cumulator(df)
    if normalize:
        qini = qini.div(np.abs(qini.iloc[-1, :]), axis=1)

    return qini


def auuc_score(df, outcome='y', treatment='x', true_effect=None, normalize=True, random_name='RANDOM', ):
    gain = get_gain(df, outcome=outcome, treatment=treatment, true_effect=true_effect,
                    normalize=normalize, random_name=random_name, )
    auuc = gain.sum() / len(gain)
    return auuc


def qini_score(df, outcome='y', treatment='x', true_effect=None, normalize=True, random_name='RANDOM'):
    qini = get_qini(df, outcome=outcome, treatment=treatment,
                    true_effect=true_effect,
                    normalize=normalize,
                    random_name='RANDOM' if random_name is None else random_name)

    if random_name is None:
        qini = (qini.sum(axis=0) - qini['RANDOM'].sum()) / len(qini)
        qini = qini.drop('RANDOM')
    else:
        qini = (qini.sum(axis=0) - qini[random_name].sum()) / len(qini)

    return qini


def plot_gain(df, outcome='y', treatment='x', true_effect=None, normalize=True,
              random_name='RANDOM', n_sample=100, **kwargs):
    gain = get_gain(df, outcome=outcome, treatment=treatment, true_effect=true_effect,
                    normalize=normalize, random_name=random_name, )
    if n_sample is not None and n_sample < len(gain):
        gain = gain.iloc[np.linspace(0, df.index[-1], n_sample, endpoint=True)]

    gain.plot(**kwargs)


def plot_qini(df, outcome='y', treatment='x', true_effect=None, normalize=True,
              n_sample=100, **kwargs):
    qini = get_qini(df, outcome=outcome, treatment=treatment,
                    true_effect=true_effect,
                    normalize=normalize,
                    random_name='RANDOM')

    if n_sample is not None and n_sample < len(qini):
        qini = qini.iloc[np.linspace(0, df.index[-1], n_sample, endpoint=True)]

    qini.plot(**kwargs)
