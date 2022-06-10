import math
from functools import partial

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ylearn import sklearn_ex as skex
from ylearn.causal_discovery import DagDiscovery
from ylearn.causal_model import CausalModel, CausalGraph
from ylearn.effect_interpreter.policy_interpreter import PolicyInterpreter
from ylearn.estimator_model.base_models import BaseEstLearner
from ylearn.policy.policy_model import PolicyTree
from ylearn.utils import logging, view_pydot, infer_task_type, to_repr, drop_none
from ._factory import ESTIMATOR_FACTORIES

logger = logging.get_logger(__name__)

GRAPH_STRING_BASE = """
 digraph G {
  graph [splines=true pad=0.5]
  node [shape="box" width=3 height=1.2]
 
  W [label="Adjustments\n\n WLIST" pos="5,2!" width=5]
  V [label="Covariates\n\n VLIST"  pos="10,2!"]
  X [label="Treatments\n\n XLIST"  pos="5,0!"]
  Y [label="Outcome\n\n YLIST"  pos="10,0!"] 
 
  W -> {X Y}  
  V -> {X Y}
  X -> Y
}
"""

GRAPH_STRING_IV = """
 digraph G {
  graph [splines=true pad=0.5]
  node [shape="box" width=3 height=1.2]

  W [label="Adjustments\n\n WLIST" pos="7,2!" width=5]
  Z [label="Instruments\n\n ZLIST" pos="3,0!"]
  X [label="Treatments\n\n XLIST"  pos="7,0!"  width=2]
  Y [label="Outcome\n\n YLIST"  pos="10,0!" width=2] 

  W -> {X Y}
  Z -> X -> Y
}
"""


def _to_list(v, name=None):
    if v is None or isinstance(v, (list, tuple)):
        pass
    elif isinstance(v, str):
        v = [s.strip() for s in v.split(',')]
        v = [s for s in v if len(s) > 0]
    else:
        tag = name if name is not None else 'value'
        raise ValueError(f'Unexpected {tag}: {v}')

    return v


def _join_list(*args):
    r = []
    for a in args:
        if a is None:
            pass
        elif isinstance(a, list):
            r += a
        elif isinstance(a, tuple):
            r += list(a)
        else:
            r += _to_list(a)
    return r


def _safe_remove(alist, value, copy=False):
    assert alist is None or isinstance(alist, list)

    if alist is not None:
        if copy:
            alist = alist.copy()

        if isinstance(value, (list, tuple)):
            for v in value:
                if v in alist:
                    alist.remove(v)
        elif value in alist:
            alist.remove(value)

    return alist


def _format(v, line_width=64, line_limit=3):
    if isinstance(v, (list, tuple)):
        lines = []
        line = ''
        for vi in v:
            if len(line) >= line_width:
                if len(lines) + 1 >= line_limit:
                    line += '...'
                    break
                else:
                    lines.append(line)
                    line = ''  # reset new line
            line += f', {vi}' if line else f'{vi}'
        lines.append(line)
        r = ',\n'.join(lines)
    else:
        r = f'{v}'

    return r


def _is_number(dtype):
    return dtype.kind in {'i', 'f'}


class CausalConsole:
    # def __init__(self, feature_inds, categorical, heterogeneity_inds=None, feature_names=None, classification=False,
    #              upper_bound_on_cat_expansion=5, nuisance_models='linear', heterogeneity_model='linear', *,
    #              categories='auto', n_jobs=-1, verbose=0, cv=5, mc_iters=3, skip_cat_limit_checks=False,
    #              random_state=None):
    def __init__(self,
                 task=None,  # str, default infer from outcome
                 identify_method='auto',  # discovery, feature_importances, ...
                 discovery_model=None,  # notears_linear, ...
                 estimator='auto',  # auto, dml, dr, ml or metaleaner, iv, div, ...
                 estimator_options=None,  # dict or None
                 scorer=None,  # auto, rloss, or None
                 scorer_options=None,  # dict or None
                 random_state=None):
        assert isinstance(estimator, (str, BaseEstLearner))
        if isinstance(estimator, str):
            assert estimator == 'auto' or estimator in ESTIMATOR_FACTORIES.keys()
        assert scorer is None or scorer in {'auto', 'rloss'}

        self.identify_method = identify_method
        self.discovery_model = discovery_model
        self.estimator = estimator
        self.estimator_options = estimator_options
        self.scorer = scorer
        self.scorer_options = scorer_options
        self.random_state = random_state

        # fitted
        self.feature_names_in_ = None
        self.task = task
        self.treatment_ = None
        self.outcome_ = None
        self.adjustment_ = None
        self.covariate_ = None
        self.instrument_ = None

        self.causation_matrix_ = None  # if discovery_model='discovery'

        #
        self.y_encoder_ = None
        self.preprocessor_ = None
        self.estimators_ = None
        self.scorers_ = None
        # ...

    # def fit(self, X, y, warm_start=False):
    def fit(self, data,
            outcome,  # required, str, one
            *,
            treatment=None,  # str list or int, one or more
            adjustment=None,  # feature names, default infer by feature importances
            covariate=None,  # ??
            instrument=None,  # ??
            # treat=None,
            # control=None,
            treatment_count_limit=None,
            copy=True,
            **kwargs
            ):
        """
        steps:
            * infer task type if None
            * discovery causal graph
            * identify adjustment with identification_model
            * fit causal estimator with nuisance_models and heterogeneity_model

        """
        assert isinstance(data, pd.DataFrame)
        feature_names = [c for c in data.columns.tolist()]

        if copy:
            data = data.copy()

        y = data[outcome]
        if self.task is None:
            self.task, _ = infer_task_type(y)
            logger.info(f'infer task as {self.task}')

        if not _is_number(y.dtype):
            logger.info('encode outcome with LabelEncoder')
            y_encoder = LabelEncoder()
            y = y_encoder.fit_transform(y)
            data[outcome] = y
        else:
            y_encoder = None

        logger.info(f'identify treatment, adjustment, covariate and instrument')
        treatment, adjustment, covariate, instrument = self._identify(
            data, outcome,
            treatment=treatment, adjustment=adjustment, covariate=covariate, instrument=instrument,
            treatment_count_limit=treatment_count_limit,
        )

        ##
        logger.info('preprocess data ...')
        preprocessor = skex.general_preprocessor()
        columns = _join_list(adjustment, covariate, instrument)
        assert len(columns) > 0

        data_t = preprocessor.fit_transform(data[columns], y)
        assert isinstance(data_t, pd.DataFrame) and set(data_t.columns.tolist()) == set(columns)
        data[columns] = data_t

        estimators = {}
        scorers = {} if self.scorer is not None else None
        for x in treatment:
            estimator = self._create_estimator(
                data, outcome, treatment=x,
                adjustment=adjustment, covariate=covariate, instrument=instrument)

            logger.info(f'fit estimator for {x} with {estimator}')
            fit_kwargs = dict(**drop_none(adjustment=adjustment, covariate=covariate, instrument=instrument),
                              **kwargs)
            estimator.fit(data, outcome, x, **fit_kwargs)
            estimators[x] = estimator

            if self.scorer is not None:
                scorer = self._create_scorer(
                    data, outcome, treatment=x,
                    adjustment=adjustment, covariate=covariate, instrument=instrument
                )
                logger.info(f'fit scorer for {x} with {scorer}')
                scorer.fit(data, outcome, x,
                           **drop_none(adjustment=adjustment, covariate=covariate, instrument=instrument)
                           )
                scorers[x] = scorer

        self.feature_names_in_ = feature_names
        self.outcome_ = outcome
        self.treatment_ = treatment
        self.adjustment_ = adjustment
        self.covariate_ = covariate
        self.instrument_ = instrument

        self.estimators_ = estimators
        self.scorers_ = scorers
        self.y_encoder_ = y_encoder
        self.preprocessor_ = preprocessor

        return self

    @staticmethod
    def _discovery(data, outcome, random_state):
        logger.info('discovery causation')

        X = data.copy()
        y = X.pop(outcome)

        if not _is_number(y.dtype):
            y = LabelEncoder().fit_transform(y)

        preprocessor = skex.general_preprocessor()
        X = preprocessor.fit_transform(X, y)
        X[outcome] = y

        dd = DagDiscovery(random_state=random_state)
        return dd(data)

    def _identify(
            self, data, outcome, *,
            treatment=None, adjustment=None, covariate=None, instrument=None,
            treatment_count_limit=None):

        if treatment is None:
            treatment = self._identify_treatment(data, outcome, treatment_count_limit)

        treatment = _to_list(treatment, name='treatment')
        adjustment = _to_list(adjustment, name='adjustment')
        covariate = _to_list(covariate, name='covariate')
        instrument = _to_list(instrument, name='instrument')

        _safe_remove(adjustment, treatment)
        _safe_remove(covariate, treatment)
        _safe_remove(instrument, treatment)

        if all(map(lambda a: a is None or len(a) == 0, (adjustment, covariate, instrument))):
            adjustment, covariate, instrument = self._identify_aci(data, outcome, treatment)

        return treatment, adjustment, covariate, instrument

    def _identify_aci(self, data, outcome, treatment):
        adjustment, covariate, instrument = [], [], []

        if self.identify_method == 'discovery':
            if self.causation_matrix_ is None:
                self.causation_matrix_ = self._discovery(data, outcome, self.random_state)
            causation = self.causation_matrix_
            threshold = causation.values.diagonal().max()
            m = DagDiscovery().matrix2dict(causation, threshold=threshold)
            cg = CausalGraph(m)
            cm = CausalModel(cg)
            try:
                instrument = cm.get_iv(treatment[0], outcome)  # fixme
                instrument = [c for c in instrument if c != outcome and c not in treatment]
            except Exception as e:
                logger.warn(e)

            if len(instrument) == 0:
                ids = cm.identify(treatment, outcome, identify_method=('backdoor', 'simple'))
                covariate = list(set(ids['backdoor'][0]))
            else:
                covariate = [c for c in data.columns.tolist()
                             if c != outcome and c not in treatment and c not in instrument]
        else:
            covariate = [c for c in data.columns.tolist() if c != outcome and c not in treatment]

        return adjustment, covariate, instrument

    def _identify_treatment(self, data, outcome, adjustment=None, covariate=None, instrument=None,
                            count_limit=None):
        excludes = _join_list(adjustment, covariate, instrument)

        if count_limit is None:
            count_limit = min(math.ceil(data.shape[1] * 0.1), 5)

        if self.identify_method == 'discovery':
            fn = self._identify_treatment_by_discovery
        else:
            fn = self._identify_treatment_by_importance

        return fn(data, outcome, count_limit=count_limit, excludes=excludes)

    def _identify_treatment_by_discovery(self, data, outcome, count_limit, excludes=None):
        causation = self._discovery(data, outcome, self.random_state)
        assert isinstance(causation, pd.DataFrame) and outcome in causation.columns.tolist()

        treatment = causation[outcome].abs().sort_values(ascending=False)
        treatment = [i for i in treatment.index if treatment[i] > 0]
        if excludes is not None:
            treatment = [t for t in treatment if t not in excludes]

        if len(treatment) > count_limit:
            treatment = treatment[:count_limit]

        self.causation_matrix_ = causation

        return treatment

    def _identify_treatment_by_importance(self, data, outcome, count_limit, excludes=None):
        X = data.copy()
        y = X.pop(outcome)

        if excludes is not None and len(excludes) > 0:
            X = X[[c for c in X.columns.tolist() if c not in excludes]]

        tf = skex.FeatureImportancesSelectionTransformer(
            task=self.task, strategy='number', number=count_limit, data_clean=False)
        tf.fit(X, y)
        treatment = tf.selected_features_

        return treatment

    def _create_estimator(self, data, outcome, *,
                          treatment=None, adjustment=None, covariate=None, instrument=None):
        estimator = self.estimator
        if estimator == 'auto':
            if instrument is not None and len(instrument) > 0:
                estimator = 'iv'
            else:
                estimator = 'dml' if covariate is not None else 'ml'  # FIXME, check treatment is category or not

        x_task, _ = infer_task_type(data[treatment])
        options = self.estimator_options if self.estimator_options is not None else {}
        factory = ESTIMATOR_FACTORIES[estimator](**options)

        estimator = factory(data, outcome, y_task=self.task, x_task=x_task,
                            treatment=treatment,
                            adjustment=adjustment,
                            covariate=covariate,
                            instrument=instrument,
                            random_state=self.random_state)
        return estimator

    def _create_scorer(self, data, outcome, *,
                       treatment=None, adjustment=None, covariate=None, instrument=None):
        scorer = self.scorer
        if scorer == 'auto':
            scorer = 'rloss'

        x_task, _ = infer_task_type(data[treatment])
        options = self.scorer_options if self.scorer_options is not None else {}
        factory = ESTIMATOR_FACTORIES[scorer](**options)
        scorer = factory(data, outcome, y_task=self.task, x_task=x_task,
                         treatment=treatment,
                         adjustment=adjustment,
                         covariate=covariate,
                         instrument=instrument,
                         random_state=self.random_state)
        return scorer

    def causal_graph(self):
        causation = self.causation_matrix_
        if causation is not None:
            threshold = causation.values.diagonal().max()
            m = DagDiscovery().matrix2dict(causation, threshold=threshold)
        else:
            m = {}
            fmt = partial(_format, line_limit=1, line_width=20)
            label_y = f'Outcome({self.outcome_})'
            label_x = f'Treatments({fmt(self.treatment_)})'

            if self.adjustment_ is not None and len(self.adjustment_) > 0:
                label_w = f'Adjustments({fmt(self.adjustment_)})'
                m[label_w] = [label_x, label_y]
            if self.covariate_ is not None and len(self.covariate_) > 0:
                label_v = f'Covariates({fmt(self.covariate_)})'
                m[label_v] = [label_x, label_y]
            if self.instrument_ is not None and len(self.instrument_) > 0:
                label_z = f'Instruments({fmt(self.instrument_)})'
                m[label_z] = [label_x, ]

        cg = CausalGraph(m)
        return cg

    def causal_effect(self):
        return self.cohort_causal_effect(None)

    def cohort_causal_effect(self, Xtest):
        if Xtest is not None and self.preprocessor_ is not None:
            columns = _join_list(self.adjustment_, self.covariate_, self.instrument_)
            assert len(columns) > 0
            Xtest[columns] = self.preprocessor_.transform(Xtest[columns])

        dfs = []
        for x, est in self.estimators_.items():
            effect = est.estimate(data=Xtest)
            s = pd.Series(dict(mean=effect.mean(),
                               min=effect.min(),
                               max=effect.max(),
                               std=effect.std()),
                          name=x)
            dfs.append(s)
        return pd.concat(dfs, axis=1).T

    def local_causal_effect(self, Xtest):
        if Xtest is not None and self.preprocessor_ is not None:
            columns = _join_list(self.adjustment_, self.covariate_, self.instrument_)
            assert len(columns) > 0
            Xtest[columns] = self.preprocessor_.transform(Xtest[columns])

        dfs = []
        for x, est in self.estimators_.items():
            effect = est.estimate(data=Xtest)
            s = pd.Series(effect.ravel(), name=x)
            dfs.append(s)
        return pd.concat(dfs, axis=1)

    def whatif(self, data, new_value, treatment=None):
        assert data is not None and new_value is not None
        assert treatment is None or isinstance(treatment, str)
        if isinstance(treatment, str):
            assert treatment in self.treatment_
        if treatment is None:
            treatment = self.treatment_[0]

        estimator = self.estimators_[treatment]
        if estimator.is_discrete_treatment:
            return self._whatif_discrete(data, new_value, treatment, estimator)
        else:
            return self._whatif_continuous(data, new_value, treatment, estimator)

    def _whatif_discrete(self, data, new_value, treatment, estimator):
        y_old = data[self.outcome_]
        old_value = data[treatment]

        df = pd.DataFrame(dict(c=old_value, t=new_value), index=old_value.index)
        df['tc'] = df[['t', 'c']].apply(tuple, axis=1)
        tc_pairs = df['tc'].unique().tolist()
        effect = []
        for t, c in tc_pairs:
            tc_rows = df[df['tc'] == (t, c)]
            if t == c:
                eff = np.zeros((len(tc_rows),))
            else:
                data_rows = data.loc[tc_rows.index]
                eff = estimator.estimate(data_rows, treat=t, control=c)
            effect.append(pd.DataFrame(dict(e=eff.ravel()), index=tc_rows.index))
        effect = pd.concat(effect, axis=0)
        assert len(effect) == len(df)
        df['e'] = effect['e']  # align indices

        y_new = y_old + df['e']
        return y_new

    def _whatif_continuous(self, data, new_value, treatment, estimator):
        y_old = data[self.outcome_]
        old_value = data[treatment]
        effect = estimator.estimate(data, treat=new_value, control=old_value)
        y_new = y_old + effect.ravel()
        return y_new

    def score(self, Xtest=None):
        assert Xtest is None  # fixme

        if self.scorers_ is None:
            raise ValueError(f'scorer was disabled. setup scorer and fit the {type(self).__init__} pls.')

        sa = []
        for x, scorer in self.scorers_.items():
            est = self.estimators_[x]
            sa.append(scorer.score(est))
        score = np.mean(sa)

        return score

    def policy_tree(self, Xtest, treatment=None, **tree_options):
        if treatment is None:
            treatment = self.treatment_[0]
        estimator = self.estimators_[treatment]

        tree_options = dict(criterion='policy_reg', **tree_options)
        ptree = PolicyTree(**tree_options)
        ptree.fit(Xtest, covariate=self.covariate_, est_model=estimator)

        return ptree

    def policy_interpreter(self, data, treatment=None, **options):
        if treatment is None:
            treatment = self.treatment_[0]
        estimator = self.estimators_[treatment]

        pi = PolicyInterpreter(**options)
        pi.fit(data, estimator)

        return pi

    def plot_policy_tree(self, Xtest, treatment=None, treat=None, control=None, **tree_options):
        ptree = self.policy_tree(Xtest, treatment=treatment, treat=treat, control=control, **tree_options)
        ptree.plot()

    def plot_policy_interpreter(self, data, treatment=None, options=None, **kwargs):
        pi = self.policy_interpreter(data, treatment=treatment, options=options)
        pi.plot(**kwargs)

    def plot_causal_graph(self):
        import pydot

        values = dict(WLIST=self.adjustment_, VLIST=self.covariate_, XLIST=self.treatment_,
                      YLIST=self.outcome_, ZLIST=self.instrument_)
        dot_string = GRAPH_STRING_BASE if self.instrument_ is None else GRAPH_STRING_IV
        for k, v in values.items():
            if dot_string.find(k) >= 0:
                width = 40 if k == 'ZLIST' else 64
                dot_string = dot_string.replace(k, _format(v, line_width=width))
        graph = pydot.graph_from_dot_data(dot_string)[0]
        view_pydot(graph, prog='fdp')

    # def plot_heterogeneity_tree(self, Xtest, feature_index, *,
    #                             max_depth=3, min_samples_leaf=2, min_impurity_decrease=1e-4,
    #                             include_model_uncertainty=False,
    #                             alpha=0.05):
    def plot_heterogeneity_tree(self, Xtest, treatment=None, **tree_options):
        raise NotImplemented()

    # # ??
    # def individualized_policy(self, Xtest, treatment=None,
    #                           *,
    #                           n_rows=None, treatment_costs=0, alpha=0.05):
    #     pass

    def __repr__(self):
        return to_repr(self)
