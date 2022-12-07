import copy
import inspect
from itertools import product

import numpy as np
import pandas as pd
from joblib import delayed, Parallel

from .base_models import BaseEstModel
from .doubly_robust import DoublyRobust
from .meta_learner import SLearner, TLearner, XLearner

try:
    from .causal_tree import CausalTree
except ImportError as e:
    _msg_causal_tree = f'{e}'


    class CausalTree:
        def __init__(self, *args, **kwargs):
            raise ImportError(_msg_causal_tree)


def _default_estimate_options(learner, effect_nji=False):
    options = {}
    if learner.is_discrete_outcome:
        if effect_nji:
            fn = learner.effect_nji
        else:
            fn = learner.estimate
        params = inspect.signature(fn).parameters
        if 'target_outcome' in params.keys():
            classes = getattr(learner, 'outcome_classes_', None)
            if isinstance(classes, (list, tuple, np.ndarray)):
                options['target_outcome'] = classes[-1]
    return options


def _copy_and_fit(learner, data, outcome, treatment, treat, control, **kwargs):
    learner = copy.deepcopy(learner)
    learner.fit(data, outcome, treatment, treat=treat, control=control, **kwargs)
    return learner


def _effect_nji(learner_and_sign, data, outcome, treatment, treat, control, target_outcome):
    y_count = 1 if isinstance(outcome, str) else len(outcome)

    if treat == control:
        effect = np.zeros((len(data), y_count, y_count))
    else:
        assert isinstance(learner_and_sign, tuple)

        learner, sign = learner_and_sign
        effect = learner.effect_nji(data)
        assert len(effect.shape) == 3
        assert effect.shape[2] == 2

        classes = getattr(learner, 'outcome_classes_', None) if learner.is_discrete_outcome else None
        if classes is not None:
            assert effect.shape[1] == len(classes)
            if target_outcome is not None:
                assert isinstance(classes, (list, tuple, np.ndarray))
                nz = np.nonzero(np.array(classes) == target_outcome)[0]
                if len(nz) == 0:
                    assert ValueError(f'Invalid target_outcome: "{target_outcome}"')
                outcome_idx = nz[0]
            else:
                outcome_idx = len(classes) - 1
            effect = effect[:, outcome_idx:outcome_idx + 1, :]
        else:
            assert effect.shape[1] == y_count

        if np.all(effect[:, :, 1:] == 0.0):
            effect = effect[:, :, :1]
        else:
            effect = effect[:, :, 1:]
        if sign < 0:
            effect = effect * sign

    return effect


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

        # others
        self._last_treat_control = None

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

    def _get_learner(self, treat, control, silent=False):
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
            if not silent:
                raise ValueError(f'Not found learner for treat-control pair: [{treat},{control}]')
            else:
                return None

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

    def estimate(self, data=None, treat=None, control=None, target_outcome=None, **kwargs):
        treatment = self.treatment
        if isinstance(treatment, str):
            treatment = [treatment]
        if len(treatment) == 1:
            if isinstance(control, (tuple, list)):
                control = control[0]
            if isinstance(treat, (tuple, list)):
                treat = treat[0]

        learner, sign = self._get_learner(treat, control)
        if self.is_discrete_outcome:
            options = _default_estimate_options(learner)
            if target_outcome is not None:
                options['target_outcome'] = target_outcome
            options.update(kwargs)
        else:
            options = kwargs
        effect = learner.estimate(data, **options)
        if sign < 0:
            effect = effect * sign

        self._last_treat_control = (treat, control)
        return effect

    def effect_nji(self, data=None, control=None, target_outcome=None, n_jobs=None, **kwargs):
        treatment = self.treatment
        if isinstance(treatment, str):
            treatment = [treatment]

        if control is None:
            control = [self.treats_[t][0] for t in treatment]

        if isinstance(control, list):
            control = tuple(control)

        if len(treatment) == 1:
            if isinstance(control, tuple):
                control = control[0]
            treats = self.treats_[treatment[0]]
        else:
            treats = product(*[self.treats_[x] for x in treatment])

        if n_jobs in {0, 1}:
            # estimate learners one by one
            effects = [_effect_nji(
                self._get_learner(treat, control, silent=True),
                data, self.outcome, self.treatment, treat, control, target_outcome)
                for treat in treats]
        else:
            # estimate learners with joblib
            job_options = self._get_job_options(n_jobs)
            effects = Parallel(**job_options)(delayed(_effect_nji)(
                self._get_learner(treat, control, silent=True),
                data, self.outcome, self.treatment, treat, control, target_outcome
            ) for treat in treats)

        effects = np.concatenate(effects, axis=2)
        return effects

    @staticmethod
    def _get_job_options(n_jobs=None):
        return dict(n_jobs=n_jobs, prefer='processes')

    @property
    def treat(self):
        tc = self._last_treat_control
        if tc is None:
            raise ValueError('Not found the last_treat_control, call estimate firstly please.')

        learner, _ = self._get_learner(tc[0], tc[1])
        return learner.treat

    def __repr__(self):
        return f'Permuted{self.learner}'


class PermutedSLearner(PermutedLearner):
    def __init__(self, model,
                 is_discrete_treatment=True,
                 is_discrete_outcome=False,
                 categories="auto",
                 proba_output=None,
                 random_state=2022,
                 **kwargs):
        if proba_output is None:
            proba_output = is_discrete_outcome
        learner = SLearner(model,
                           random_state=random_state,
                           is_discrete_treatment=is_discrete_treatment,
                           is_discrete_outcome=is_discrete_outcome,
                           categories=categories,
                           proba_output=proba_output,
                           **kwargs)
        super().__init__(learner)


class PermutedTLearner(PermutedLearner):
    def __init__(self, model,
                 is_discrete_treatment=True,
                 is_discrete_outcome=False,
                 proba_output=None,
                 categories="auto",
                 random_state=2022,
                 **kwargs):
        if proba_output is None:
            proba_output = is_discrete_outcome
        learner = TLearner(model,
                           random_state=random_state,
                           is_discrete_treatment=is_discrete_treatment,
                           is_discrete_outcome=is_discrete_outcome,
                           categories=categories,
                           proba_output=proba_output,
                           **kwargs)
        super().__init__(learner)


class PermutedXLearner(PermutedLearner):
    def __init__(self, model,
                 final_proba_model=None,
                 is_discrete_treatment=True,
                 is_discrete_outcome=False,
                 proba_output=None,
                 categories="auto",
                 random_state=2022,
                 **kwargs):
        if proba_output is None:
            proba_output = is_discrete_outcome
        learner = XLearner(model,
                           random_state=random_state,
                           is_discrete_treatment=is_discrete_treatment,
                           is_discrete_outcome=is_discrete_outcome,
                           proba_output=proba_output,
                           final_proba_model=final_proba_model,
                           categories=categories,
                           **kwargs)
        super().__init__(learner)


class PermutedDoublyRobust(PermutedLearner):
    def __init__(self, x_model, y_model, yx_model, *args, **kwargs):
        learner = DoublyRobust(x_model, y_model, yx_model, *args, **kwargs)
        super().__init__(learner)


class PermutedCausalTree(PermutedLearner):
    def __init__(self, **kwargs):
        learner = CausalTree(**kwargs)
        super().__init__(learner)
