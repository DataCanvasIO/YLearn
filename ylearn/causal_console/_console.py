import copy as _copy

import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from ylearn import sklearn_ex as skex
from ylearn.estimator_model.double_ml import DML4CATE
from ylearn.utils import view_pydot, infer_task_type, to_repr, const, logging

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

TASK_ESTIMATOR_MAPPING = {
    # y_task - x_task : estimator
    f'{const.TASK_REGRESSION}-{const.TASK_REGRESSION}': DML4CATE(
        y_model=RandomForestRegressor(), x_model=RandomForestRegressor(), is_discrete_treatment=False),
    f'{const.TASK_REGRESSION}-{const.TASK_BINARY}': DML4CATE(
        y_model=RandomForestRegressor(), x_model=RandomForestClassifier(), is_discrete_treatment=True),
    f'{const.TASK_REGRESSION}-{const.TASK_MULTICLASS}': DML4CATE(
        y_model=RandomForestRegressor(), x_model=RandomForestClassifier(), is_discrete_treatment=True),
    f'{const.TASK_BINARY}-{const.TASK_REGRESSION}': DML4CATE(
        y_model=RandomForestClassifier(), x_model=RandomForestRegressor(), is_discrete_treatment=False),
    f'{const.TASK_BINARY}-{const.TASK_BINARY}': DML4CATE(
        y_model=RandomForestClassifier(), x_model=RandomForestClassifier(), is_discrete_treatment=True),
    f'{const.TASK_BINARY}-{const.TASK_MULTICLASS}': DML4CATE(
        y_model=RandomForestClassifier(), x_model=RandomForestClassifier(), is_discrete_treatment=True),
}


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


class CausalConsole:
    # def __init__(self, feature_inds, categorical, heterogeneity_inds=None, feature_names=None, classification=False,
    #              upper_bound_on_cat_expansion=5, nuisance_models='linear', heterogeneity_model='linear', *,
    #              categories='auto', n_jobs=-1, verbose=0, cv=5, mc_iters=3, skip_cat_limit_checks=False,
    #              random_state=None):
    def __init__(self,
                 task=None,  # str, default infer from outcome
                 identify_method='auto',  # discovery, feature_importances, ...
                 discovery_model=None,  # notears_linear, ...
                 treatment_count_limit=None,  # int or float
                 estimator='auto',  # auto, ...
                 random_state=None):
        self.identify_method = identify_method
        self.discovery_model = discovery_model
        self.treatment_count_limit = treatment_count_limit
        self.estimator = estimator
        self.random_state = random_state

        # fitted
        self.task = task
        self.treatment_ = None
        self.outcome_ = None
        self.adjustment_ = None
        self.covariate_ = None
        self.instrument_ = None

        #
        self.y_encoder_ = None
        self.preprocessor_ = None
        self.estimators_ = None
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
        if copy:
            data = data.copy()

        treatment, adjustment, covariate, instrument = self._identify(
            data, outcome,
            treatment=treatment, adjustment=adjustment, covariate=covariate, instrument=instrument
        )
        self.outcome_ = outcome
        self.treatment_ = treatment
        self.adjustment_ = adjustment
        self.covariate_ = covariate
        self.instrument_ = instrument

        ##
        preprocessor = skex.general_preprocessor()

        y = data.pop(outcome)

        if self.task is None:
            self.task, _ = infer_task_type(y)
            logger.info(f'infer task as {self.task}')

        if self.task in (const.TASK_BINARY, const.TASK_MULTICLASS):
            logger.info('encode outcome with LabelEncoder')
            y_encoder = LabelEncoder()
            y = y_encoder.fit_transform(y)
        else:
            y_encoder = None

        logger.info('preprocess data ...')
        data_t = preprocessor.fit_transform(data, y)
        data_t[outcome] = y
        estimators = {}

        for x in treatment:
            estimator = self._create_estimator(
                data_t, outcome, treatment=x,
                adjustment=adjustment, covariate=covariate, instrument=instrument)

            logger.info(f'fit estimator for {x}  as {estimator}')
            estimator.fit(data_t, outcome, x, adjustment=adjustment, covariate=covariate)
            estimators[x] = estimator

        self.estimators_ = estimators
        self.y_encoder_ = y_encoder
        self.preprocessor_ = preprocessor
        return self

    def _identify(
            self, data, outcome, *,
            treatment=None, adjustment=None, covariate=None, instrument=None):
        if treatment is None:
            treatment = self._identify_treatment(data, outcome)

        treatment = _to_list(treatment, name='treatment')
        adjustment = _to_list(adjustment, name='adjustment')
        covariate = _to_list(covariate, name='covariate')
        instrument = _to_list(instrument, name='instrument')

        _safe_remove(adjustment, treatment)
        _safe_remove(covariate, treatment)
        _safe_remove(instrument, treatment)

        return treatment, adjustment, covariate, instrument

    def _identify_treatment(self, data, outcome):
        # FIXME
        assert self.identify_method == 'auto'

        return self._identify_treatment_by_importance(data, outcome)

    def _identify_treatment_by_importance(self, data, outcome):
        X = data.copy()
        y = X.pop(outcome)
        n = self.treatment_count_limit
        if n is None:
            n = 0.1
        tf = skex.FeatureImportancesSelectionTransformer(task=self.task, strategy='number', number=n)
        tf.fit(X, y)
        treatment = tf.selected_features_

        return treatment

    def _create_estimator(self, data, outcome, *,
                          treatment=None, adjustment=None, covariate=None, instrument=None):
        # FIXME
        assert self.estimator == 'auto'
        
        t_task, _ = infer_task_type(data[treatment])
        key = f'{self.task}-{t_task}'
        assert key in TASK_ESTIMATOR_MAPPING.keys(), f'Not found estimator for {key}.'

        estimator = _copy.deepcopy(TASK_ESTIMATOR_MAPPING[key])
        for attr in ['random_state', ]:
            if hasattr(estimator, attr):
                setattr(estimator, attr, getattr(self, attr))
        return estimator

    def causal_graph(self):
        raise NotImplemented()

    def plot_graph(self):
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

    def causal_effect(self):
        return self.cohort_causal_effect(None)

    def cohort_causal_effect(self, Xtest):
        if Xtest is not None and self.preprocessor_ is not None:
            Xtest = self.preprocessor_.transform(Xtest)

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
            Xtest = self.preprocessor_.transform(Xtest)

        dfs = []
        for x, est in self.estimators_.items():
            effect = est.estimate(data=Xtest)
            s = pd.Series(effect.ravel(), name=x)
            dfs.append(s)
        return pd.concat(dfs, axis=1)

    def whatif(self, data, new_value, treatment):
        raise NotImplemented()

    def score(self, Xtest):
        raise NotImplemented()

    # def plot_heterogeneity_tree(self, Xtest, feature_index, *,
    #                             max_depth=3, min_samples_leaf=2, min_impurity_decrease=1e-4,
    #                             include_model_uncertainty=False,
    #                             alpha=0.05):
    def plot_heterogeneity_tree(self, Xtest, treatment=None, **tree_options):
        raise NotImplemented()

    def plot_policy_tree(self, Xtest, treatment=None, **tree_options):
        raise NotImplemented()

    # # ??
    # def individualized_policy(self, Xtest, treatment=None,
    #                           *,
    #                           n_rows=None, treatment_costs=0, alpha=0.05):
    #     pass

    def __repr__(self):
        return to_repr(self)
