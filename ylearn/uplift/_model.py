import numpy as np
import pandas as pd

from ylearn.utils import check_fitted as _check_fitted
from ._metric import get_gain, get_qini, get_cumlift, auuc_score, qini_score
from ._plot import plot_gain, plot_qini, plot_cumlift


class UpliftModel(object):
    def __init__(self):
        # fitted
        self.lift_ = None
        self.cumlift_ = None
        self.gain_ = None
        self.gain_top_point_ = None
        self.qini_ = None
        self.qini_top_point_ = None
        self.auuc_score_ = None
        self.qini_score_ = None
        self.random_ = None

    def fit(self, df_lift, outcome='y', treatment='x', true_effect=None, treat=1, control=0, random='RANDOM'):
        assert isinstance(df_lift, pd.DataFrame)

        # self.df_lift_ = df_lift

        self.cumlift_ = get_cumlift(
            df_lift, outcome=outcome, treatment=treatment, true_effect=true_effect,
            treat=treat, control=control, random_name=random)
        self.gain_, self.gain_top_point_ = get_gain(
            df_lift, outcome=outcome, treatment=treatment, true_effect=true_effect,
            treat=treat, control=control, random_name=random, return_top_point=True,
            normalize=False)
        self.qini_, self.qini_top_point_ = get_qini(
            df_lift, outcome=outcome, treatment=treatment, true_effect=true_effect,
            treat=treat, control=control, random_name=random, return_top_point=True,
            normalize=False)
        self.auuc_score_ = auuc_score(
            df_lift, outcome=outcome, treatment=treatment, true_effect=true_effect,
            treat=treat, control=control, random_name=random,
            normalize=True
        )
        self.qini_score_ = qini_score(
            df_lift, outcome=outcome, treatment=treatment, true_effect=true_effect,
            treat=treat, control=control, random_name=random,
            normalize=True
        )
        self.random_ = random
        self.lift_ = df_lift.copy()
        self._is_fitted = True

        return self

    @_check_fitted
    def get_cumlift(self):
        return self.cumlift_.copy()

    @_check_fitted
    def get_gain(self, normalize=False):
        gain = self.gain_
        if normalize:
            gain = gain.div(np.abs(gain.iloc[-1, :]), axis=1)

        return gain.copy()

    @_check_fitted
    def get_qini(self, normalize=False):
        qini = self.qini_
        if normalize:
            qini = qini.div(np.abs(qini.iloc[-1, :]), axis=1)

        return qini.copy()

    def _get_value_or_series(self, s, name=None):
        assert isinstance(s, pd.Series)
        if name is None:
            if self.random_ is not None:
                return s[s.index != self.random_].copy()
            else:
                return s.copy()
        else:
            return s[name]

    @_check_fitted
    def gain_top_point(self, name=None):
        return self._get_value_or_series(self.gain_top_point_, name)

    @_check_fitted
    def qini_top_point(self, name=None):
        return self._get_value_or_series(self.qini_top_point_, name)

    @_check_fitted
    def auuc_score(self, name=None):
        return self._get_value_or_series(self.auuc_score_, name)

    @_check_fitted
    def qini_score(self, name=None):
        return self._get_value_or_series(self.qini_score_, name)

    @_check_fitted
    def plot_qini(self, n_sample=100, normalize=False, **kwargs):
        plot_qini(self.get_qini(normalize=normalize), n_sample=n_sample, **kwargs)

    @_check_fitted
    def plot_gain(self, n_sample=100, normalize=False, **kwargs):
        plot_gain(self.get_gain(normalize=normalize), n_sample=n_sample, **kwargs)

    @_check_fitted
    def plot_cumlift(self, n_bins=10, **kwargs):
        plot_cumlift(self.get_cumlift(), n_bins=n_bins, **kwargs)
