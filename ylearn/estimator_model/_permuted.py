import copy
from itertools import product

import pandas as pd
from joblib import delayed, Parallel

from .base_models import BaseEstModel
from .doubly_robust import DoublyRobust
from .meta_learner import SLearner, TLearner, XLearner


def _copy_and_fit(learner, data, outcome, treatment, treat, control, **kwargs):
    learner = copy.deepcopy(learner)
    learner.fit(data, outcome, treatment, treat=treat, control=control, **kwargs)
    return learner


class PermutedLearner(BaseEstModel):
    def __init__(self, learner):
        assert isinstance(learner, BaseEstModel) and learner.is_discrete_treatment
        assert not learner._is_fitted

        super().__init__(random_state=learner.random_state,
                         is_discrete_treatment=learner.is_discrete_treatment,
                         is_discrete_outcome=learner.is_discrete_outcome,
                         _is_fitted=False,
                         # is_discrete_instrument=False,
                         categories=learner.categories)

        self.learner = learner

        # fitted
        self.learners_ = {}
        self._is_fitted = False

    def _permute_treats(self):
        treatment = self.treatment
        if isinstance(treatment, str):
            treatment = [treatment]
        assert isinstance(treatment, (list, tuple)) and len(treatment) > 0

        single_treatment = len(treatment) == 1
        if single_treatment:
            permuted_treats = self.treats_[treatment[0]]
        else:
            permuted_treats = [self.treats_[t] for t in treatment]
            permuted_treats = tuple(product(*permuted_treats))

        n = len(permuted_treats)
        for t in range(1, n):
            for c in range(t):
                if single_treatment:
                    yield permuted_treats[t], permuted_treats[c]
                else:
                    yield tuple(permuted_treats[t]), tuple(permuted_treats[c])

    def _get_learner(self, treat, control):
        assert self._is_fitted

        treatment = self.treatment
        if isinstance(treatment, str):
            treatment = [treatment]
        assert isinstance(treatment, (list, tuple)) and len(treatment) > 0

        single_treatment = len(treatment) == 1

        if treat is None:
            treat = [self.treats_[t][-1] for t in treatment]
            if single_treatment:
                treat = treat[0]

        if control is None:
            control = [self.treats_[t][0] for t in treatment]
            if single_treatment:
                control = control[0]

        if isinstance(treat, list):
            treat = tuple(treat)
        if isinstance(control, list):
            control = tuple(control)

        if (treat, control) in self.learners_.keys():
            return self.learners_[(treat, control)], 1
        elif (control, treat) in self.learners_.keys():
            return self.learners_[(control, treat)], -1
        else:
            raise ValueError(f'Not found leaner for treat-control pair: [{treat},{control}]')

    def fit(
            self,
            data,
            outcome,
            treatment,
            # adjustment=None,
            # covariate=None,
            n_jobs=None,
            **kwargs,
    ):
        assert self.is_discrete_treatment
        assert isinstance(data, pd.DataFrame)

        super().fit(data, outcome, treatment, **kwargs)

        learners = {}
        if n_jobs in {0, 1}:
            # fit learners one by one
            for treat, control in self._permute_treats():
                learners[(treat, control)] = _copy_and_fit(
                    self.learner, data, outcome, treatment, treat, control, **kwargs
                )
        else:
            # fit learners with joblib
            job_options = self._get_job_options(n_jobs)
            tc_pairs = list(self._permute_treats())
            ls = Parallel(**job_options)(delayed(_copy_and_fit)(
                self.learner, data, outcome, treatment, t, c, **kwargs
            ) for t, c in tc_pairs)
            for (t, c), l in zip(tc_pairs, ls):
                learners[(t, c)] = l

        self.learners_ = learners
        self._is_fitted = True
        return self

    def estimate(self, data=None, treat=None, control=None, **kwargs):
        learner, sign = self._get_learner(treat, control)
        effect = learner.estimate(data, **kwargs)
        if sign < 0:
            effect = effect * sign
        return effect

    def effect_nji(self, data=None, treat=None, control=None, **kwargs):
        learner, sign = self._get_learner(treat, control)
        effect = learner.effect_nji(data, **kwargs)
        if sign < 0:
            effect = effect * sign
        return effect

    @staticmethod
    def _get_job_options(n_jobs=None):
        return dict(n_jobs=n_jobs, prefer='processes')


class PermutedSLearner(PermutedLearner):
    def __init__(self, model, *args, **kwargs):
        learner = SLearner(model, *args, **kwargs)
        super().__init__(learner)


class PermutedTLearner(PermutedLearner):
    def __init__(self, model, *args, **kwargs):
        learner = TLearner(model, *args, **kwargs)
        super().__init__(learner)


class PermutedXLearner(PermutedLearner):
    def __init__(self, model, *args, **kwargs):
        learner = XLearner(model, *args, **kwargs)
        super().__init__(learner)


class PermutedDoublyRobust(PermutedLearner):
    def __init__(self, x_model, y_model, yx_model, *args, **kwargs):
        learner = DoublyRobust(x_model, y_model, yx_model, *args, **kwargs)
        super().__init__(learner)
