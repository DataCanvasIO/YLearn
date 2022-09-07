import numpy as np
import pandas as pd


def _sample(df, n_sample, reset_index=True):
    assert isinstance(df, pd.DataFrame) and len(df) > n_sample

    idx = np.linspace(0, len(df) - 1, n_sample, endpoint=True, dtype='int')
    result = df.iloc[idx]
    if reset_index:
        # result = result.reset_index(drop=True)
        result.index = idx
    return result


def _split(df_cumlift, n_bins):
    dfs = []
    for x in df_cumlift.columns.tolist():
        # if x == 'RANDOM':
        #     continue  # ignore it
        df = df_cumlift[[x]].copy()
        df['_k_'] = pd.qcut(df[x], n_bins, labels=np.arange(0, n_bins, 1))
        df = df.groupby(by='_k_')[x].mean().sort_index(ascending=False)
        df.index = pd.RangeIndex(0, n_bins)  # np.arange(0, bins, 1)
        df.name = x
        dfs.append(df)

    result = pd.concat(dfs, axis=1) if len(dfs) > 1 else dfs[0]
    return result


def plot_cumlift(*cumlift, n_bins=10, **kwargs):
    assert all(isinstance(e, pd.DataFrame) or hasattr(e, 'get_cumlift') for e in cumlift)

    dfs = [e if isinstance(e, pd.DataFrame) else e.get_cumlift() for e in cumlift]
    dfs = [_split(df, n_bins) for df in dfs]
    df = pd.concat(dfs, axis=1) if len(dfs) > 1 else dfs[0]

    options = dict(rot=0, ylabel='cumlift')
    options.update(kwargs)
    df.plot.bar(**options)


def plot_gain(*gain, n_sample=100, normalize=False, **kwargs):
    assert all(isinstance(e, pd.DataFrame) or hasattr(e, 'get_gain') for e in gain)

    dfs = [e if isinstance(e, pd.DataFrame) else e.get_gain(normalize=normalize) for e in gain]
    dfs = [_sample(df, n_sample) for df in dfs]
    df = pd.concat(dfs, axis=1) if len(dfs) > 1 else dfs[0]

    options = dict(ylabel='Gain', xlabel='Population', grid=True)
    options.update(kwargs)
    df.plot(**options)


def plot_qini(*qini, n_sample=100, normalize=False, **kwargs):
    assert all(isinstance(e, pd.DataFrame) or hasattr(e, 'get_qini') for e in qini)

    dfs = [e if isinstance(e, pd.DataFrame) else e.get_qini(normalize=normalize) for e in qini]
    dfs = [_sample(df, n_sample) for df in dfs]
    df = pd.concat(dfs, axis=1) if len(dfs) > 1 else dfs[0]

    options = dict(ylabel='Qini', xlabel='Population', grid=True)
    options.update(kwargs)
    df.plot(**options)
