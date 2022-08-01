import math
from functools import partial
from itertools import product

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ylearn import metric as M
from ylearn import sklearn_ex as skex
from ylearn.causal_discovery import CausalDiscovery
from ylearn.causal_model import CausalModel, CausalGraph
from ylearn.effect_interpreter.policy_interpreter import PolicyInterpreter
from ylearn.estimator_model import ESTIMATOR_FACTORIES, BaseEstModel
from ylearn.utils import const, logging, view_pydot, infer_task_type, to_repr, drop_none, set_random_state

logger = logging.get_logger(__name__)

DEFAULT_TREATMENT_COUNT_LIMIT_PERCENT = 0.1
DEFAULT_TREATMENT_COUNT_LIMIT_BOUND = 5

GRAPH_STRING_BASE = """
 digraph G {
  graph [splines=true pad=0.5]
  node [shape="box" width=3 height=1.2]

  V [label="Covariates\n\n VLIST"  pos="7.5,2!"]
  X [label="Treatments\n\n XLIST"  pos="5,0!"]
  Y [label="Outcome\n\n YLIST"  pos="10,0!"] 

  V -> {X Y}
  X -> Y
}
"""

GRAPH_STRING_W = """
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

GRAPH_STRING_Z = """
 digraph G {
  graph [splines=true pad=0.5]
  node [shape="box" width=3 height=1.2]

  V [label="Covariates\n\n VLIST" pos="7,2!" width=5]
  Z [label="Instruments\n\n ZLIST" pos="3,0!"]
  X [label="Treatments\n\n XLIST"  pos="7,0!"  width=2]
  Y [label="Outcome\n\n YLIST"  pos="10,0!" width=2] 

  V -> {X Y}
  Z -> X -> Y
}
"""

GRAPH_STRING_WZ = """
 digraph G {
  graph [splines=true pad=0.5]
  node [shape="box" width=3 height=1.2]

  W [label="Adjustments\n\n WLIST" pos="5,2!" width=5]
  V [label="Covariates\n\n VLIST" pos="10,2!" width=5]
  Z [label="Instruments\n\n ZLIST" pos="3,0!"]
  X [label="Treatments\n\n XLIST"  pos="7,0!"  width=2]
  Y [label="Outcome\n\n YLIST"  pos="10,0!" width=2] 

  W -> {X Y}
  V -> {X Y}
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


def _empty(v):
    return v is None or len(v) == 0


def _not_empty(v):
    return v is not None and len(v) > 0


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


def _is_discrete(y):
    return infer_task_type(y)[0] != const.TASK_REGRESSION


def _task_tag(discrete):
    if isinstance(discrete, (np.ndarray, pd.Series)):
        discrete = _is_discrete(discrete)
    return 'classification' if discrete else 'regression'


def _align_task_to_first(data, columns, count_limit):
    """
     select features which has similar task type with the first
    """
    selected = columns[:1]
    discrete = _is_discrete(data[selected[0]])
    for x in columns[1:]:
        x_discrete = _is_discrete(data[x])
        if x_discrete is discrete:
            selected.append(x)
            if len(selected) >= count_limit:
                break

    return selected


def _select_by_task(data, columns, count_limit, discrete):
    """
     select features which has similar task type with the first
    """
    selected = []
    for x in columns:
        x_discrete = _is_discrete(data[x])
        if x_discrete is discrete:
            selected.append(x)
            if len(selected) >= count_limit:
                break

    return selected


class Identifier:
    def identify_treatment(self, data, outcome, discrete_treatment, count_limit, excludes=None):
        raise NotImplemented()

    def identify_aci(self, data, outcome, treatment):
        raise NotImplemented()


class DefaultIdentifier(Identifier):
    def identify_treatment(self, data, outcome, discrete_treatment, count_limit, excludes=None):
        X = data.copy()
        y = X.pop(outcome)

        if excludes is not None and len(excludes) > 0:
            X = X[[c for c in X.columns.tolist() if c not in excludes]]

        tf = skex.FeatureImportancesSelectionTransformer(
            strategy='number', number=X.shape[1], data_clean=False)
        tf.fit(X, y)
        selected = tf.selected_features_

        if discrete_treatment is None:
            treatment = _align_task_to_first(data, selected, count_limit)
        else:
            treatment = _select_by_task(data, selected, count_limit, discrete_treatment)

        return treatment

    def identify_aci(self, data, outcome, treatment):
        adjustment, instrument = None, None

        covariate = [c for c in data.columns.tolist() if c != outcome and c not in treatment]

        return adjustment, covariate, instrument


class IdentifierWithDiscovery(DefaultIdentifier):
    def __init__(self, random_state, **kwargs):
        self.random_state = random_state
        self.discovery_options = kwargs.copy()
        self.causation_matrix_ = None

    def _discovery(self, data, outcome):
        logger.info('discovery causation')

        X = data.copy()
        y = X.pop(outcome)

        if not _is_number(y.dtype):
            y = LabelEncoder().fit_transform(y)

        # preprocessor = skex.general_preprocessor(number_scaler=True)
        preprocessor = skex.general_preprocessor()
        X = preprocessor.fit_transform(X, y)
        X[outcome] = y

        options = dict(random_state=self.random_state)
        if self.discovery_options is not None:
            options.update(self.discovery_options)

        dd = CausalDiscovery(**options)
        return dd(X)

    def identify_treatment(self, data, outcome, discrete_treatment, count_limit, excludes=None):
        causation = self._discovery(data, outcome)
        assert isinstance(causation, pd.DataFrame) and outcome in causation.columns.tolist()

        treatment = causation[outcome].abs().sort_values(ascending=False)
        treatment = [i for i in treatment.index if treatment[i] > 0 and i != outcome]
        if excludes is not None:
            treatment = [t for t in treatment if t not in excludes]

        if len(treatment) > 0:
            if discrete_treatment is None:
                treatment = _align_task_to_first(data, treatment, count_limit)
            else:
                treatment = _select_by_task(data, treatment, count_limit, discrete_treatment)
        else:
            logger.info(f'Not found treatment with causal discovery, so identify treatment by default')
            treatment = super().identify_treatment(data, outcome, discrete_treatment, count_limit, excludes=excludes)

        self.causation_matrix_ = causation

        return treatment

    def identify_aci(self, data, outcome, treatment):
        adjustment, covariate = None, None

        if self.causation_matrix_ is None:
            self.causation_matrix_ = self._discovery(data, outcome)
        causation = self.causation_matrix_
        # threshold = causation.values.diagonal().max()
        threshold = min(np.quantile(causation.values.diagonal(), 0.8),
                        np.mean(causation.values))

        if np.isnan(threshold):
            return super().identify_aci(data, outcome, treatment)

        m = CausalDiscovery().matrix2dict(causation, threshold=threshold)
        cg = CausalGraph(m)
        cm = CausalModel(cg)
        try:
            instrument = cm.get_iv(treatment[0], outcome)
            if _not_empty(instrument):
                instrument = [c for c in instrument if c != outcome and c not in treatment]
            for x in treatment[1:]:
                if _empty(instrument):
                    break
                iv = cm.get_iv(x, outcome)
                if _empty(iv):
                    instrument = None
                    break
                else:
                    iv = [c for c in iv if c != outcome and c not in treatment]
                    instrument = list(set(instrument).intersection(set(iv)))
            if logger.is_info_enabled():
                logger.info(f'found instrument: {instrument}')
        except Exception as e:
            logger.warn(e)
            instrument = []

        if _empty(instrument):
            ids = cm.identify(treatment, outcome, identify_method=('backdoor', 'simple'))
            covariate = list(set(ids['backdoor'][0]))
            covariate = [c for c in covariate if c != outcome and c not in treatment]

        if _empty(covariate):
            logger.info('Not found covariate by discovery, so setup it by default')
            covariate = [c for c in data.columns.tolist()
                         if c != outcome and c not in treatment and c not in instrument]
        else:
            logger.info(f'found covariate: {covariate}')

        if _empty(instrument):
            instrument = None

        return adjustment, covariate, instrument


class Why:
    """
    An all-in-one API for causal learning.

    Parameters
    ----------
    discrete_outcome : bool, default=None
        If True, force the outcome as discrete;
        If False, force the outcome as continuous;
        If None, inferred from outcome.
    discrete_treatment : bool, default=None
        If True, force the treatment variables as discrete;
        If False, force the treatment variables as continuous;
        if None, inferred from the first treatment
    identifier : str, default='auto'
        Available options: 'auto' or 'discovery'
    discovery_model : str, default=None
        Reserved
    discovery_options : dict, default=None
        Parameters (key-values) to initialize the discovery model
    estimator : str, default='auto'
        Name of a valid EstimatorModel. One can also pass an instance of a valid estimator model
    estimator_options : dict, default=None
        Parameters (key-values) to initialize the estimator model.
    random_state : int, default=None
        Random state seed

    Attributes
    ----------
    feature_names_in_ : list of feature names seen during `fit`
    outcome_ : name of outcome
    treatment_ : list of treatment names identified during `fit`
    adjustment_ : list of adjustment names identified during `fit`
    covariate_ : list of covariate names identified during `fit`
    instrument_ : list of instrument names identified during `fit`
    identifier_ : identifier object or None
         Used to identify treatment/adjustment/covariate/instrument if they were not specified during `fit`
    y_encoder_ : LabelEncoder object or None
        Used to encode outcome if it is discrete
    x_encoders_ : LabelEncoder dict for each treatment if discrete_treatment is True
        Key is the treatment name, value is the LabelEncoder object
        None if discrete_treatment is False
    preprocessor_ : Pipeline object to preprocess data during `fit`
    estimators_ : estimators dict for each treatment
        Key is the treatment name, value is the estimator object

    """

    def __init__(self,
                 discrete_outcome=None,
                 discrete_treatment=None,
                 identifier='auto',
                 discovery_model=None,
                 discovery_options=None,
                 estimator='auto',
                 estimator_options=None,
                 random_state=None):
        assert isinstance(estimator, (str, BaseEstModel))
        if isinstance(estimator, str):
            assert estimator == 'auto' or estimator in ESTIMATOR_FACTORIES.keys()

        self.discrete_outcome = discrete_outcome
        self.discrete_treatment = discrete_treatment
        self.identifier = identifier
        self.discovery_model = discovery_model
        self.discovery_options = discovery_options
        self.estimator = estimator
        self.estimator_options = estimator_options
        self.random_state = random_state

        # fitted
        self._is_fitted = False,
        self.feature_names_in_ = None
        self.treatment_ = None
        self.outcome_ = None
        self.adjustment_ = None
        self.covariate_ = None
        self.instrument_ = None

        #
        self.identifier_ = None
        self.y_encoder_ = None
        self.x_encoders_ = None
        self.preprocessor_ = None
        self.estimators_ = None

    def fit(self, data, outcome,
            *,
            treatment=None,
            adjustment=None,
            covariate=None,
            instrument=None,
            treatment_count_limit=None,
            copy=True,
            **kwargs
            ):
        """
        Fit the Why object, steps:
            1. encode outcome if its dtype is not numeric
            2. identify treatment and adjustment/covariate/instrument
            3. encode treatment if discrete_treatment is True
            4. preprocess data
            5. fit causal estimators

        Parameters
        ----------
        data : pandas.DataFrame, required
        outcome : str, required
            Outcome feature name
        treatment : str or list of str
            Names of the treatment variables.
            If str, will be split into list with comma;
            if None, identified by identifier.
        adjustment : str or list of str, optional
            Names of the adjustment variables. Identified by identifier if adjustment/covariate/instrument are all None.
            If str, will be split into list with comma
        covariate : str or list of str, optional
            Names of the covariate variables. Identified by identifier if adjustment/covariate/instrument are all None.
            If str, will be split into list with comma
        instrument : str or list of str, optional
            Names of the instrument variables. Identified by identifier if adjustment/covariate/instrument are all None.
            If str, will be split into list with comma
        treatment_count_limit : int, default None
            Maximum treatment number, default `min(5, 10% of total feature number)`.
        copy : bool, default True
            Set to False to perform inplace transforming and avoid a copy of data.
        kwargs : options to fit estimators

        Returns
        -------
        instance of :py:class:`Why`
            fitted Why object
        """
        assert isinstance(data, pd.DataFrame)

        feature_names = [c for c in data.columns.tolist()]

        assert outcome is not None and isinstance(outcome, str) and len(outcome) > 0
        assert outcome in feature_names

        set_random_state(self.random_state)
        if copy:
            data = data.copy()

        y = data[outcome]
        if self.discrete_outcome is None:
            self.discrete_outcome = _is_discrete(y)
            logger.info(f'infer outcome as {_task_tag(self.discrete_outcome)}')

        # if not _is_number(y.dtype):
        if self.discrete_outcome:
            logger.info('encode outcome with LabelEncoder')
            y_encoder = LabelEncoder()
            y = y_encoder.fit_transform(y)
            data[outcome] = y
        else:
            y_encoder = None

        logger.info(f'identify treatment, adjustment, covariate and instrument')
        treatment, adjustment, covariate, instrument = self.identify(
            data, outcome,
            treatment=treatment, adjustment=adjustment, covariate=covariate, instrument=instrument,
            treatment_count_limit=treatment_count_limit,
        )

        # encode treatment
        if self.discrete_treatment:
            logger.info(f'encode treatment ...')
            x_encoders = {}
            for x in treatment:
                x_encoder = LabelEncoder()
                data[x] = x_encoder.fit_transform(data[x])
                x_encoders[x] = x_encoder
        else:
            x_encoders = None

        # preprocess adjustment, covariate, instrument
        logger.info('preprocess data ...')
        preprocessor = skex.general_preprocessor()
        columns = _join_list(adjustment, covariate, instrument)
        assert len(columns) > 0
        data_t = preprocessor.fit_transform(data[columns], y)
        assert isinstance(data_t, pd.DataFrame) and set(data_t.columns.tolist()) == set(columns)
        data[columns] = data_t

        # fit estimator
        estimator_keys = treatment.copy()
        if self.discrete_treatment and len(treatment) > 1:
            estimator_keys.extend(list(filter(lambda _: _[0] != _[1], product(treatment, treatment))))
        estimators = {}
        for x in estimator_keys:
            estimator = self._create_estimator(
                data, outcome, treatment=x,
                adjustment=adjustment, covariate=covariate, instrument=instrument)

            logger.info(f'fit estimator for {x} with {estimator}')
            fit_kwargs = dict(**drop_none(adjustment=adjustment, covariate=covariate, instrument=instrument),
                              **kwargs)
            if isinstance(x, tuple):
                estimator.fit(data, outcome, list(x), **fit_kwargs)
            else:
                estimator.fit(data, outcome, x, **fit_kwargs)
            estimators[x] = estimator

        # save state
        self.feature_names_in_ = feature_names
        self.outcome_ = outcome
        self.treatment_ = treatment
        self.adjustment_ = adjustment
        self.covariate_ = covariate
        self.instrument_ = instrument

        self.estimators_ = estimators

        self.y_encoder_ = y_encoder
        self.x_encoders_ = x_encoders
        self.preprocessor_ = preprocessor
        self._is_fitted = True

        return self

    @staticmethod
    def _get_default_treatment_count_limit(data, outcome):
        return min(math.ceil(data.shape[1] * DEFAULT_TREATMENT_COUNT_LIMIT_PERCENT),
                   DEFAULT_TREATMENT_COUNT_LIMIT_BOUND)

    def _get_default_estimator(self, data, outcome, options, *,
                               treatment, adjustment=None, covariate=None, instrument=None):
        if instrument is not None and len(instrument) > 0:
            estimator = 'iv'
        # elif x_task in {const.TASK_BINARY, const.TASK_MULTICLASS}:
        elif self.discrete_treatment:
            estimator = 'ml'
        else:
            estimator = 'dml'

        return estimator, options

    def identify(self, data, outcome, *,
                 treatment=None, adjustment=None, covariate=None, instrument=None, treatment_count_limit=None):
        """
        Identify treatment and adjustment/covariate/instrument without fitting Why.

        Parameters
        ----------
        data : pandas.DataFrame, required
        outcome : str, outcome feature name, required
        treatment : str or list of str
            Names of the treatment variables.
            If str, will be split into list with comma;
            if None, identified by identifier
        adjustment : str or list of str, optional
            Names of the adjustment variables. Identified by identifier if adjustment/covariate/instrument are all None.
            If str, will be split into list with comma
        covariate : str or list of str, optional
            Names of the covariate variables. Identified by identifier if adjustment/covariate/instrument are all None.
            If str, will be split into list with comma
        instrument : str or list of str, optional
            Names of the instrument variables. Identified by identifier if adjustment/covariate/instrument are all None.
            If str, will be split into list with comma
        treatment_count_limit : int, default None
            Maximum treatment number, default `min(5, 10% of total feature number)`.
        Returns
        -------
        tuple of identified treatment, adjustment, covariate, instrument
        """

        identifier = None

        treatment = _to_list(treatment, name='treatment')
        if _empty(treatment):
            identifier = self._create_identifier()
            if treatment_count_limit is None:
                treatment_count_limit = self._get_default_treatment_count_limit(data, outcome)
            treatment = identifier.identify_treatment(data, outcome, self.discrete_treatment, treatment_count_limit)
            if logger.is_info_enabled():
                tag = 'classification' if _is_discrete(data[treatment[0]]) else 'regression'
                logger.info(f'identified treatment[{tag}]: {treatment}')
        else:
            x_tasks = dict(map(lambda x: (x, 'classification' if _is_discrete(data[x]) else 'regression'),
                               treatment))
            if len(set(x_tasks.values())) > 1:
                logger.warn('Both regression and classification features are used as treatment, '
                            'something maybe unexpected.'
                            f'\n{x_tasks}')

        if self.discrete_treatment is None:
            self.discrete_treatment = _is_discrete(data[treatment[0]])
            if logger.is_info_enabled():
                logger.info(f'infer discrete_treatment={self.discrete_treatment}')

        adjustment = _to_list(adjustment, name='adjustment')
        covariate = _to_list(covariate, name='covariate')
        instrument = _to_list(instrument, name='instrument')

        _safe_remove(adjustment, treatment)
        _safe_remove(covariate, treatment)
        _safe_remove(instrument, treatment)

        if all(map(_empty, (adjustment, covariate, instrument))):
            if identifier is None:
                identifier = self._create_identifier()
            adjustment, covariate, instrument = identifier.identify_aci(data, outcome, treatment)
            if logger.is_info_enabled():
                logger.info(f'identified adjustment: {adjustment}')
                logger.info(f'identified covariate: {covariate}')
                logger.info(f'identified instrument: {instrument}')

        self.identifier_ = identifier
        return treatment, adjustment, covariate, instrument

    def _create_identifier(self):
        if self.identifier == 'discovery':
            options = self.discovery_options if self.discovery_options is not None else {}
            return IdentifierWithDiscovery(self.random_state, **options)
        else:
            return DefaultIdentifier()

    def _create_estimator(self, data, outcome, *,
                          treatment, adjustment=None, covariate=None, instrument=None):
        # x_task, _ = infer_task_type(data[treatment])
        estimator = self.estimator
        options = self.estimator_options if self.estimator_options is not None else {}
        if estimator == 'auto':
            estimator, options = self._get_default_estimator(
                data, outcome, options,
                treatment=treatment,
                adjustment=adjustment,
                covariate=covariate,
                instrument=instrument,
            )

        factory = ESTIMATOR_FACTORIES[estimator](**options)

        estimator = factory(data, outcome, y_task=self.discrete_outcome, x_task=self.discrete_treatment,
                            treatment=treatment,
                            adjustment=adjustment,
                            covariate=covariate,
                            instrument=instrument,
                            random_state=self.random_state)
        return estimator

    def _create_scorers(self, data, scorer):
        if isinstance(scorer, BaseEstModel):
            return {x: scorer for x in self.treatment_}

        if scorer is None or scorer == 'auto':
            scorer = 'rloss'

        factory = ESTIMATOR_FACTORIES[scorer]()
        scorers = {}
        for x in self.treatment_:
            # x_task, _ = infer_task_type(data[x])
            scorer = factory(data, self.outcome_,
                             y_task=self.discrete_outcome,
                             x_task=self.discrete_treatment,
                             treatment=x,
                             adjustment=self.adjustment_,
                             covariate=self.covariate_,
                             instrument=self.instrument_,
                             random_state=self.random_state)
            scorers[x] = scorer
        return scorers

    def _preprocess(self, test_data, encode_outcome=True, encode_treatment=False):
        assert self._is_fitted

        if test_data is not None:
            test_data = test_data.copy()
            columns = test_data.columns.tolist()

            if self.preprocessor_ is not None:
                var_columns = _join_list(self.adjustment_, self.covariate_, self.instrument_)
                assert len(var_columns) > 0
                test_data[var_columns] = self.preprocessor_.transform(test_data[var_columns])

            if encode_outcome and self.y_encoder_ is not None and self.outcome_ in columns:
                test_data[self.outcome_] = self.y_encoder_.transform(test_data[self.outcome_])

            if encode_treatment and self.x_encoders_ is not None:
                for x, xe in self.x_encoders_.items():
                    if x in columns:
                        test_data[x] = xe.transform(test_data[x])

        return test_data

    def _safe_treat_control(self, t_or_c, name):
        assert self._is_fitted

        if t_or_c is None:
            return None

        if isinstance(t_or_c, (tuple, list)):
            assert len(t_or_c) == len(self.treatment_), \
                f'{name} should have the same number with treatment ({self.treatment_})'
        else:
            assert len(self.treatment_) == 1, \
                f'{name} should be list or tuple if the number of treatment is greater than 1'
            t_or_c = [t_or_c, ]

        return t_or_c

    def causal_graph(self):
        """
        Get identified causal graph

        Returns
        -------
        CausalGraph object
        """
        causation = self.identifier_.causation_matrix_ \
            if isinstance(self.identifier_, IdentifierWithDiscovery) else None

        if causation is not None:
            threshold = causation.values.diagonal().max()
            m = CausalDiscovery().matrix2dict(causation, threshold=threshold)
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

    def causal_effect(self, test_data=None, treat=None, control=None, return_detail=False):
        """
        Estimate the causal effect.

        Parameters
        ----------
        test_data : pd.DataFrame, default None
            The test data to evaluate the causal effect.
            If None, the training data is used.
        treat : treatment value or list or ndarray or pandas.Series, default None
            In the case of single discrete treatment, treat should be an int or
            str of one of all possible treatment values which indicates the
            value of the intended treatment;
            in the case of multiple discrete treatment, treat should be a list
            where treat[i] indicates the value of the i-th intended treatment,
            for example, when there are multiple discrete treatments,
            list(['run', 'read']) means the treat value of the first treatment is taken as 'run'
            and that of the second treatment is taken as 'read';
            in the case of continuous treatment, treat should be a float or a ndarray or pandas.Series,
            by default None
        control : treatment value or list or ndarray or pandas.Series, default None
            This is similar to the cases of treat, by default None
        return_detail: bool, default False
            If True, return effect details in result.

        Returns
        -------
        pd.DataFrame
            causal effect of each treatment. The result DataFrame columns are:
                mean: mean of causal effect,
                min: minimum of causal effect,
                max: maximum of causal effect,
                detail(if return_detail is True ): causal effect ndarray;
            In the case of discrete treatment, the result DataFrame indices are multiindex of
            (treatment name and treat_vs_control);
            in the case of continuous treatment, the result DataFrame indices are treatment names.

        """
        test_data = self._preprocess(test_data)
        treat = self._safe_treat_control(treat, 'treat')
        control = self._safe_treat_control(control, 'control')

        if self.discrete_treatment:
            return self._causal_effect_discrete(test_data, treat, control, return_detail=return_detail)
        else:
            return self._causal_effect_continuous(test_data, treat, control, return_detail=return_detail)

    def _causal_effect_discrete(self, test_data=None, treat=None, control=None, return_detail=False):
        dfs = []
        for i, x in enumerate(self.treatment_):
            est = self.estimators_[x]
            xe = self.x_encoders_[x]
            if control is not None:
                assert control[i] in xe.classes_.tolist(), f'Invalid {x} control "{control[i]}"'
                control_i = control[i]
            else:
                control_i = xe.classes_[0]

            if treat is not None:
                assert treat[i] in xe.classes_.tolist(), f'Invalid {x} treat "{treat[i]}"'
                treats = [treat[i], ]
            else:
                if test_data is not None and x in test_data.columns.tolist():
                    treats = np.unique(test_data[x]).tolist()
                else:
                    treats = xe.classes_.tolist()
                treats = filter(lambda _: _ != control_i, treats)

            if test_data is not None and x in test_data.columns.tolist():
                test_data[x] = xe.transform(test_data[x])
            c = xe.transform([control_i])[0]
            for treat_i in treats:
                t = xe.transform([treat_i])[0]
                effect = est.estimate(data=test_data, treat=t, control=c)
                s = pd.Series(dict(mean=effect.mean(),
                                   min=effect.min(),
                                   max=effect.max(),
                                   std=effect.std(),
                                   ),
                              name=(x, f'{treat_i} vs {control_i}')
                              )
                if return_detail:
                    s['detail'] = effect.ravel()
                dfs.append(s)

        result = pd.concat(dfs, axis=1).T
        return result

    def _causal_effect_continuous(self, test_data=None, treat=None, control=None, return_detail=False):
        dfs = []
        for i, x in enumerate(self.treatment_):
            est = self.estimators_[x]
            treat_i = treat[i] if treat is not None else None
            control_i = control[i] if control is not None else None
            if isinstance(treat_i, (pd.Series, pd.DataFrame)):
                treat_i = treat_i.values
            if isinstance(control_i, (pd.Series, pd.DataFrame)):
                control_i = control_i.values
            if treat_i is not None and control_i is None:
                control_i = np.zeros_like(treat_i)
            elif treat_i is None and control_i is not None:
                treat_i = np.ones_like(control_i)
            effect = est.estimate(data=test_data, treat=treat_i, control=control_i)
            s = pd.Series(dict(mean=effect.mean(),
                               min=effect.min(),
                               max=effect.max(),
                               std=effect.std()),
                          name=x)
            if return_detail:
                s['detail'] = effect.ravel()
            dfs.append(s)
        return pd.concat(dfs, axis=1).T

    def individual_causal_effect(self, test_data, control=None):
        """
        Estimate the causal effect for each individual.

        Parameters
        ----------
        test_data : pd.DataFrame, required
            The test data to evaluate the causal effect.
        control : treatment value or list or ndarray or pandas.Series, default None
            In the case of single discrete treatment, control should be an int or
            str of one of all possible treatment values which indicates the
            value of the intended treatment;
            in the case of multiple discrete treatment, control should be a list
            where control[i] indicates the value of the i-th intended treatment,
            for example, when there are multiple discrete treatments,
            list(['run', 'read']) means the treat value of the first treatment is taken as 'run'
            and that of the second treatment is taken as 'read';
            in the case of continuous treatment, treat should be a float or a ndarray or pandas.Series,
            by default None

        Returns
        -------
        pd.DataFrame
            individual causal effect of each treatment.
            The result DataFrame columns are the treatment names;
            In the case of discrete treatment, the result DataFrame indices are multiindex of
            (individual index in test_data, treatment name and treat_vs_control);
            in the case of continuous treatment, the result DataFrame indices are multiindex of
            (individual index in test_data, treatment name).
        """
        assert test_data is not None

        test_data = self._preprocess(test_data)
        control = self._safe_treat_control(control, 'control')

        if self.discrete_treatment:
            return self._individual_causal_effect_discrete(test_data, control)
        else:
            return self._individual_causal_effect_continuous(test_data, control)

    def _individual_causal_effect_discrete(self, test_data, control=None):
        if control is None:
            control = [self.x_encoders_[x].classes_.tolist()[0] for x in self.treatment_]

        dfs = []
        for ri in test_data.index:
            row_df = test_data.loc[ri:ri]
            for xi, x in enumerate(self.treatment_):
                treat_i = row_df[x].tolist()[0]
                control_i = control[xi]
                if treat_i == control_i:
                    effect = 0.0
                else:
                    xe = self.x_encoders_[x]
                    row_df = row_df.copy()
                    row_df[x] = xe.transform(row_df[x])
                    t, c = xe.transform([treat_i, control_i]).tolist()
                    est = self.estimators_[x]
                    effect = est.estimate(data=row_df, treat=t, control=c).ravel()[0]
                s = pd.Series(dict(effect=effect),
                              name=(ri, x, f'{treat_i} vs {control_i}'))
                dfs.append(s)
        return pd.concat(dfs, axis=1).T

    def _individual_causal_effect_continuous(self, test_data, control=None):
        dfs = []
        for i, x in enumerate(self.treatment_):
            est = self.estimators_[x]
            treat_i = test_data[x]
            control_i = control[i] if control is not None else None
            effect = est.estimate(data=test_data, treat=treat_i, control=control_i)
            s = pd.Series(effect.ravel(), name=x)
            dfs.append(s)
        return pd.concat(dfs, axis=1)

    def whatif(self, test_data, new_value, treatment=None):
        """
        Get counterfactual predictions when treatment is changed to new_value from its observational counterpart.

        Parameters
        ----------
        test_data : pd.DataFrame, required
            The test data to predict.
        new_value : ndarray or pd.Series, required
            It should have the same length with test_data.
        treatment : str, default None
            Treatment name.
            If str, it should be on of the fitted attribute **treatment_**.
            If None, then first element in the attribute **treatment_** is used.
        Returns
        -------
        pd.Series
            The counterfactual prediction
        """
        assert test_data is not None and new_value is not None
        assert treatment is None or isinstance(treatment, str)
        if isinstance(treatment, str):
            assert treatment in self.treatment_
        if treatment is None:
            treatment = self.treatment_[0]

        data_t = self._preprocess(test_data)
        estimator = self.estimators_[treatment]
        if estimator.is_discrete_treatment:
            return self._whatif_discrete(data_t, new_value, treatment, estimator)
        else:
            return self._whatif_continuous(data_t, new_value, treatment, estimator)

    def _whatif_discrete(self, data, new_value, treatment, estimator):
        y_old = data[self.outcome_]
        old_value = data[treatment]
        xe = self.x_encoders_[treatment]

        for x in self.treatment_:
            data[x] = self.x_encoders_[x].transform(data[x])

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
                t_encoded = xe.transform([t])[0]
                c_encoded = xe.transform([c])[0]
                eff = estimator.estimate(data_rows, treat=t_encoded, control=c_encoded)
            effect.append(pd.DataFrame(dict(e=eff.ravel()), index=tc_rows.index))
        effect = pd.concat(effect, axis=0)
        assert len(effect) == len(df)
        df['e'] = effect['e']  # align indices

        y_new = y_old + df['e']
        return y_new

    def _whatif_continuous(self, data, new_value, treatment, estimator):
        y_old = data[self.outcome_]
        old_value = data[treatment].values
        if isinstance(new_value, pd.Series):
            new_value = new_value.values
        effect = estimator.estimate(data, treat=new_value, control=old_value)
        y_new = y_old + effect.ravel()
        return y_new

    def score(self, test_data=None, treat=None, control=None, scorer='auto'):
        """
        Scoring the fitted estimator models.

        Parameters
        ----------
        test_data : pd.DataFrame, required
        treat : treatment value or list or ndarray or pandas.Series, default None
            In the case of single discrete treatment, treat should be an int or
            str of one of all possible treatment values which indicates the
            value of the intended treatment;
            in the case of multiple discrete treatment, treat should be a list
            where treat[i] indicates the value of the i-th intended treatment,
            for example, when there are multiple discrete treatments,
            list(['run', 'read']) means the treat value of the first treatment is taken as 'run'
            and that of the second treatment is taken as 'read';
            in the case of continuous treatment, treat should be a float or a ndarray or pandas.Series,
            by default None
        control : int or list, default None
            This is similar to the cases of treat, by default None
        scorer: str, default 'auto'
            Reserved.

        Returns
        -------
        float
            score
        """
        assert test_data is not None

        if scorer in ['auuc', 'qini']:
            return self._score_auuc_qini(test_data, treat=treat, control=control, scorer=scorer)
        else:
            return self._score_rloss(test_data, treat=treat, control=control)

    def _score_auuc_qini(self, test_data, treat=None, control=None, scorer='auuc'):
        def scoring(effect, x, preprocessed_data):
            df = pd.DataFrame(dict(effect=effect,
                                   x=preprocessed_data[x],
                                   y=preprocessed_data[self.outcome_],
                                   ))
            if scorer == 'auuc':
                s = M.auuc_score(df, outcome='y', treatment='x', random_name=None)
            else:
                s = M.qini_score(df, outcome='y', treatment='x', random_name=None)
            return s['effect']

        self._check_x2(scorer)
        sa = self._map_effect(scoring, test_data, treat=treat, control=control)
        score = np.mean(sa)
        return score

    def _score_rloss(self, test_data=None, treat=None, control=None):
        scorers = self._create_scorers(test_data, scorer='rloss')
        test_data = self._preprocess(test_data)
        treat = self._safe_treat_control(treat, 'treat')
        control = self._safe_treat_control(control, 'control')
        fit_options = drop_none(adjustment=self.adjustment_, covariate=self.covariate_, instrument=self.instrument_)

        sa = []
        for i, (x, scorer) in enumerate(scorers.items()):
            logger.info(f'fit scorer for {x} with {scorer}')
            if self.discrete_treatment:
                xe = self.x_encoders_[x]
                test_data[x] = xe.transform(test_data[x])
                treat_i = xe.transform([treat[i], ])[0] if treat is not None else None
                control_i = xe.transform([control[i], ])[0] if control is not None else None
            else:
                treat_i = treat[i] if treat is not None else None
                control_i = control[i] if control is not None else None
            scorer.fit(test_data, self.outcome_, x, **fit_options)
            estimator = self.estimators_[x]
            score_options = drop_none(treat=treat_i, control=control_i)
            sa.append(scorer.score(estimator, **score_options))
        score = np.mean(sa)

        return score

    def _effect_array(self, preprocessed_data, treatment, control=None):
        if treatment is None:
            treatment = self.treatment_[:2]
        else:
            treatment = _to_list(treatment, 'treatment')

        if len(treatment) > 2:
            raise ValueError(f'2 treatment are supported at most.')

        if self.discrete_treatment:
            for x in self.treatment_:
                preprocessed_data[x] = self.x_encoders_[x].transform(preprocessed_data[x])

        if self.discrete_treatment and len(treatment) > 1:
            estimator = self.estimators_[tuple(treatment)]
            if control is not None:
                control = [self.x_encoders_[x].transform([control[i]])[0] for i, x in enumerate(treatment)]
            effect = estimator.effect_nji(preprocessed_data, **drop_none(control=control))
            effect_array = effect.reshape(-1, effect.shape[2])
        else:
            effects = []
            for i, x in enumerate(treatment):
                estimator = self.estimators_[x]
                if self.discrete_treatment:
                    xe = self.x_encoders_[x]
                    ci = xe.transform([control[i], ])[0] if control is not None else None
                else:
                    ci = control[i] if control is not None else None
                effect = estimator.effect_nji(preprocessed_data, **drop_none(control=ci))
                assert isinstance(effect, np.ndarray)
                assert effect.ndim == 3 and effect.shape[1] == 1
                effects.append(effect.reshape(-1, effect.shape[2]))
            effect_array = np.hstack(effects)
        return effect_array

    def policy_tree(self, test_data, treatment=None, control=None, **kwargs):
        """
        Get the policy tree

        Parameters
        ----------
        test_data : pd.DataFrame, required
        treatment: str or list, optional
            Treatment names, should be one or two element.
            default the first two elements in attribute **treatment_**
        control : treatment value or list or ndarray or pandas.Series, default None
            In the case of single discrete treatment, control should be an int or
            str of one of all possible treatment values which indicates the
            value of the intended treatment;
            in the case of multiple discrete treatment, control should be a list
            where control[i] indicates the value of the i-th intended treatment,
            for example, when there are multiple discrete treatments,
            list(['run', 'read']) means the control value of the first treatment is taken as 'run'
            and that of the second treatment is taken as 'read';
            in the case of continuous treatment, control should be a float or a ndarray or pandas.Series,
            by default None
        kwargs : dict
            options to initialize the PolicyTree

        Returns
        -------
        PolicyTree :
            The fitted PolicyTree object
        """
        from ylearn.policy.policy_model import PolicyTree

        test_data = self._preprocess(test_data)
        control = self._safe_treat_control(control, 'control')
        effect_array = self._effect_array(test_data, treatment, control=control)
        ptree = PolicyTree(**kwargs)
        ptree.fit(test_data, covariate=self.covariate_, effect_array=effect_array)

        return ptree

    def policy_interpreter(self, test_data, treatment=None, control=None, **kwargs):
        """
        Get the policy interpreter

        Parameters
        ----------
        test_data : pd.DataFrame, required
        treatment: treatment names, optional
            Should be one or two element.
            default the first two element in attribute **treatment_**
        control : treatment value or list or ndarray or pandas.Series, default None
            In the case of single discrete treatment, treat should be an int or
            str of one of all possible treatment values which indicates the
            value of the intended treatment;
            in the case of multiple discrete treatment, treat should be a list
            where treat[i] indicates the value of the i-th intended treatment,
            for example, when there are multiple discrete treatments,
            list(['run', 'read']) means the treat value of the first treatment is taken as 'run'
            and that of the second treatment is taken as 'read';
            in the case of continuous treatment, treat should be a float or a ndarray or pandas.Series,
            by default None
        kwargs : options to initialize the PolicyInterpreter

        Returns
        -------
        PolicyInterpreter :
            The fitted PolicyInterpreter object
        """
        test_data = self._preprocess(test_data)
        control = self._safe_treat_control(control, 'control')
        effect_array = self._effect_array(test_data, treatment, control=control)
        pi = PolicyInterpreter(**kwargs)
        pi.fit(test_data, covariate=self.covariate_, est_model=None, effect_array=effect_array)
        return pi

    def _check_x2(self, reason):
        e_msg = f'Only binary treatment is support for {reason}'

        assert self.discrete_treatment, e_msg
        for _, xe in self.x_encoders_.items():
            assert len(xe.classes_) == 2, e_msg

    def _map_effect(self, handler, test_data, treat=None, control=None):
        test_data = self._preprocess(test_data, encode_outcome=True, encode_treatment=True)
        treat = self._safe_treat_control(treat, 'treat')
        control = self._safe_treat_control(control, 'control')
        result = []

        for i, x in enumerate(self.treatment_):
            est = self.estimators_[x]
            if self.x_encoders_ is not None:
                xe = self.x_encoders_[x]
                classes = xe.classes_.tolist()
                if control is not None:
                    assert control[i] in classes, f'Invalid {x} control "{control[i]}" for treatment {x}'
                    control_i = control[i]
                else:
                    control_i = classes[0]

                if treat is not None:
                    assert treat[i] in classes, f'Invalid {x} treat "{treat[i]}" for treatment {x}'
                    treat_i = treat[i]
                else:
                    treat_i = classes[-1]

                t, c = xe.transform([treat_i, control_i])
            else:
                t = treat[i] if treat is not None else None
                c = control[i] if control is not None else None
            effect = est.estimate(data=test_data, treat=t, control=c)
            result.append(handler(effect, x, test_data))

        return result

    def get_gain(self, test_data, treat=None, control=None, normalize=True):
        def _get_gain(effect, x, preprocessed_data):
            df_ = pd.DataFrame({x: effect,
                                '_x_': preprocessed_data[x],
                                '_y_': preprocessed_data[self.outcome_],
                                })
            return M.get_gain(df_, outcome='_y_', treatment='_x_',
                              normalize=normalize,
                              random_name='RANDOM' if x == self.treatment_[0] else None)

        self._check_x2('get_gain')
        sa = self._map_effect(_get_gain, test_data, treat=treat, control=control)
        gain = pd.concat(sa, axis=1) if len(sa) > 1 else sa[0]
        return gain

    def get_qini(self, test_data, treat=None, control=None, normalize=True):
        def _get_gain(effect, x, preprocessed_data):
            df_ = pd.DataFrame({x: effect,
                                '_x_': preprocessed_data[x],
                                '_y_': preprocessed_data[self.outcome_],
                                })
            return M.get_qini(df_, outcome='_y_', treatment='_x_',
                              normalize=normalize,
                              random_name='RANDOM' if x == self.treatment_[0] else None)

        self._check_x2('get_qini')
        sa = self._map_effect(_get_gain, test_data, treat=treat, control=control)
        qini = pd.concat(sa, axis=1) if len(sa) > 1 else sa[0]
        return qini

    def plot_policy_tree(self, test_data, treatment=None, control=None, **kwargs):
        """
        Plot the policy tree

        Returns
        -------
        PolicyTree :
            The fitted PolicyTree object
        """
        ptree = self.policy_tree(test_data, treatment=treatment, control=control, **kwargs)
        ptree.plot()
        return ptree

    def plot_policy_interpreter(self, test_data, treatment=None, control=None, **kwargs):
        """
        Plot the policy interpreter

        Returns
        -------
        PolicyInterpreter :
            The fitted PolicyInterpreter object
        """
        pi = self.policy_interpreter(test_data, treatment=treatment, control=control, **kwargs)
        pi.plot()
        return pi

    def plot_causal_graph(self):
        """
        Plot the causal graph.
        """
        import pydot

        values = dict(WLIST=self.adjustment_, VLIST=self.covariate_, XLIST=self.treatment_,
                      YLIST=self.outcome_, ZLIST=self.instrument_)
        if not _empty(self.instrument_) and not _empty(self.adjustment_):
            dot_string = GRAPH_STRING_WZ
        elif not _empty(self.instrument_):
            dot_string = GRAPH_STRING_Z
        elif not _empty(self.adjustment_):
            dot_string = GRAPH_STRING_W
        else:
            dot_string = GRAPH_STRING_BASE

        for k, v in values.items():
            if dot_string.find(k) >= 0:
                width = 40 if k == 'ZLIST' else 64
                dot_string = dot_string.replace(k, _format(v, line_width=width))
        graph = pydot.graph_from_dot_data(dot_string)[0]
        view_pydot(graph, prog='fdp')

    def plot_gain(self, test_data, treat=None, control=None, n_sample=100, normalize=False, **kwargs):
        gain = self.get_gain(test_data, treat=treat, control=control, normalize=normalize)
        if n_sample is not None and n_sample < len(gain):
            gain = gain.iloc[np.linspace(0, gain.index[-1], n_sample, endpoint=True)]

        options = dict(ylabel='Gain', xlabel='Population', **kwargs)
        gain.plot(**options)

    def plot_qini(self, test_data, treat=None, control=None, n_sample=100, normalize=False, **kwargs):
        qini = self.get_qini(test_data, treat=treat, control=control, normalize=normalize)
        if n_sample is not None and n_sample < len(qini):
            qini = qini.iloc[np.linspace(0, qini.index[-1], n_sample, endpoint=True)]

        options = dict(ylabel='Qini', xlabel='Population', **kwargs)
        qini.plot(**options)

    def __repr__(self):
        return to_repr(self)
