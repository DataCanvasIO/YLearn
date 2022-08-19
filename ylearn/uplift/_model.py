import numpy as np

from ._metric import get_gain, get_qini, get_cumlift, auuc_score, qini_score
from ._plot import plot_gain, plot_qini, plot_cumlift


class UpliftModel(object):
    def __init__(self):
        # fitted
        # self.df_lift_ = None
        self.cumlift_ = None
        self.gain_ = None
        self.gain_top_point_ = None
        self.qini_ = None
        self.qini_top_point_ = None
        self.auuc_score_ = None
        self.qini_score_ = None

    def fit(self, df_lift, outcome='y', treatment='x', true_effect=None, treat=1, control=0, random='RANDOM'):
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
            normalize=False
        )
        self.qini_score_ = qini_score(
            df_lift, outcome=outcome, treatment=treatment, true_effect=true_effect,
            treat=treat, control=control, random_name=random,
            normalize=False
        )

        return self

    def get_gain(self, normalize=False):
        gain = self.gain_
        if normalize:
            gain = gain.div(np.abs(gain.iloc[-1, :]), axis=1)

        return gain.copy()

    def get_qini(self, normalize=False):
        qini = self.qini_
        if normalize:
            qini = qini.div(np.abs(qini.iloc[-1, :]), axis=1)

        return qini.copy()

    @property
    def auuc_score(self):
        return self.auuc_score_.values.tolist()[0]

    @property
    def qini_score(self):
        return self.qini_score_.values.tolist()[0]

    def plot_qini(self, n_sample=100, normalize=False, **kwargs):
        plot_qini(self.get_qini(normalize=normalize), n_sample=n_sample, **kwargs)

    def plot_gain(self, n_sample=100, normalize=False, **kwargs):
        plot_gain(self.get_gain(normalize=normalize), n_sample=n_sample, **kwargs)

    def plot_cumlift(self, n_bins=10, **kwargs):
        plot_cumlift(self.qini_, n_bins=n_bins, **kwargs)
