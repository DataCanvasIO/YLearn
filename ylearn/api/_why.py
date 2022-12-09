import inspect
import math
from copy import deepcopy
from functools import partial
from itertools import product

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ylearn import sklearn_ex as skex
from ylearn import uplift as L
from ylearn.causal_discovery import BaseDiscovery
from ylearn.causal_model import CausalGraph
from ylearn.estimator_model import ESTIMATOR_FACTORIES, BaseEstModel
from ylearn.utils import logging, view_pydot, to_repr, drop_none, set_random_state
from . import utils
from ._identifier import Identifier, DefaultIdentifier
from ._identifier import IdentifierWithNotears, IdentifierWithLearner, IdentifierWithDiscovery

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
    identifier : str or Identifier, default='auto'
        if str, available options: 'auto' or 'discovery' or 'gcastle' or 'pgm'
    identifier_options : dict, default=None
        Parameters (key-values) to initialize the identifier
    estimator : str, default='auto'
        Name of a valid EstimatorModel. One can also pass an instance of a valid estimator model
    estimator_options : dict, default=None
        Parameters (key-values) to initialize the estimator model.
    fn_cost : callable, default None
        Cost function,  used to readjust the causal effect based on cost.
    effect_name: str, default 'effect'
        The column name in the argument DataFrame passed to fn_cost.
        Effective when fn_cost is not None.
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
                 identifier_options=None,
                 estimator='auto',
                 estimator_options=None,
                 fn_cost=None,
                 effect_name='effect',
                 random_state=None):
        assert isinstance(identifier, (str, Identifier)) or callable(identifier)
        if isinstance(identifier, str):
            valid_identifiers = {'auto', 'discovery', 'notears', 'gcastle', 'pgm'}
            assert identifier in valid_identifiers, \
                f'identifier should be one of {valid_identifiers} or instance of Identifier or callable'

        assert isinstance(estimator, (str, BaseEstModel))
        if isinstance(estimator, str):
            assert estimator == 'auto' or estimator in ESTIMATOR_FACTORIES.keys(), \
                f'estimator should be \'auto\' or one of {list(ESTIMATOR_FACTORIES.keys())}'

        self.discrete_outcome = discrete_outcome
        self.discrete_treatment = discrete_treatment
        self.identifier = identifier
        self.identifier_options = identifier_options
        self.estimator = estimator
        self.estimator_options = estimator_options
        self.fn_cost = fn_cost
        self.effect_name = effect_name
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
            Set False to perform inplace transforming and avoid a copy of data.
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
            self.discrete_outcome = utils.is_discrete(y)
            logger.info(f'infer outcome as {utils.task_tag(self.discrete_outcome)}')

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
        columns = utils.join_list(adjustment, covariate, instrument)
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

    def _get_estimator(self, treatment):
        treatment = utils.to_list(treatment, 'treatment')
        if len(treatment) == 1:
            return self.estimators_[treatment[0]]
        else:
            return self.estimators_[tuple(treatment)]

    @staticmethod
    def _get_default_treatment_count_limit(data, outcome):
        return min(math.ceil(data.shape[1] * DEFAULT_TREATMENT_COUNT_LIMIT_PERCENT),
                   DEFAULT_TREATMENT_COUNT_LIMIT_BOUND)

    def _get_default_estimator(self, data, outcome, options, *,
                               treatment, adjustment=None, covariate=None, instrument=None):
        if instrument is not None and len(instrument) > 0:
            estimator = 'iv'
        # elif x_task in {const.TASK_BINARY, const.TASK_MULTICLASS}:
        elif self.discrete_treatment or self.discrete_outcome:
            estimator = 'tlearner'
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
            Maximum treatment number, default `min(5, 10% of the number of features)`.
        Returns
        -------
        tuple of identified treatment, adjustment, covariate, instrument
        """

        identifier = None

        treatment = utils.to_list(treatment, name='treatment')
        if utils.is_empty(treatment):
            identifier = self._create_identifier()
            if treatment_count_limit is None:
                treatment_count_limit = self._get_default_treatment_count_limit(data, outcome)
            treatment = identifier.identify_treatment(data, outcome, self.discrete_treatment, treatment_count_limit)
            if logger.is_info_enabled():
                tag = 'classification' if utils.is_discrete(data[treatment[0]]) else 'regression'
                logger.info(f'identified treatment[{tag}]: {treatment}')
        else:
            x_tasks = dict(map(lambda x: (x, 'classification' if utils.is_discrete(data[x]) else 'regression'),
                               treatment))
            if len(set(x_tasks.values())) > 1:
                logger.warn('Both regression and classification features are used as treatment, '
                            'something maybe unexpected.'
                            f'\n{x_tasks}')

        if self.discrete_treatment is None:
            self.discrete_treatment = utils.is_discrete(data[treatment[0]])
            if logger.is_info_enabled():
                logger.info(f'infer discrete_treatment={self.discrete_treatment}')

        adjustment = utils.to_list(adjustment, name='adjustment')
        covariate = utils.to_list(covariate, name='covariate')
        instrument = utils.to_list(instrument, name='instrument')

        utils.safe_remove(adjustment, treatment)
        utils.safe_remove(covariate, treatment)
        utils.safe_remove(instrument, treatment)

        if all(map(utils.is_empty, (adjustment, covariate, instrument))):
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
        if isinstance(self.identifier, Identifier):
            return self.identifier
        elif callable(self.identifier):
            options = self.identifier_options if self.identifier_options is not None else {}
            return IdentifierWithLearner(self.identifier, random_state=self.random_state, **options)
        elif self.identifier in {'discovery', 'notears'}:
            options = self.identifier_options if self.identifier_options is not None else {}
            return IdentifierWithNotears(random_state=self.random_state, **options)
        elif self.identifier == 'gcastle':
            from ._identifier import IdentifierWithGCastle
            options = self.identifier_options if self.identifier_options is not None else {}
            return IdentifierWithGCastle(random_state=self.random_state, **options)
        elif self.identifier == 'pgm':
            from ._identifier import IdentifierWithPgm
            options = self.identifier_options if self.identifier_options is not None else {}
            return IdentifierWithPgm(random_state=self.random_state, **options)
        else:
            return DefaultIdentifier()

    def _create_estimator(self, data, outcome, *,
                          treatment, adjustment=None, covariate=None, instrument=None):
        # x_task, _ = infer_task_type(data[treatment])
        estimator = self.estimator
        assert isinstance(estimator, (str, BaseEstModel))
        if isinstance(estimator, str):
            assert estimator == 'auto' or estimator in ESTIMATOR_FACTORIES.keys()

        if isinstance(estimator, BaseEstModel):
            return deepcopy(estimator)

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
                var_columns = utils.join_list(self.adjustment_, self.covariate_, self.instrument_)
                assert len(var_columns) > 0
                test_data[var_columns] = self.preprocessor_.transform(test_data[var_columns])

            if encode_outcome and self.y_encoder_ is not None and self.outcome_ in columns:
                test_data[self.outcome_] = self.y_encoder_.transform(test_data[self.outcome_])

            if encode_treatment and self.x_encoders_ is not None:
                for x, xe in self.x_encoders_.items():
                    if x in columns:
                        test_data[x] = xe.transform(test_data[x])

        return test_data

    def _safe_treat_control(self, treatment, t_or_c, name):
        assert self._is_fitted
        return utils.safe_treat_control(treatment, t_or_c, name)

    def _safe_treat_control_list(self, treatment, t_or_c, name):
        assert self._is_fitted
        return utils.safe_treat_control_list(treatment, t_or_c, name)

    def _inverse_transform_outcome(self, v):
        if self.y_encoder_ is not None and v is not None:
            return self.y_encoder_.inverse_transform([v])[0]
        else:
            return v

    def causal_graph(self):
        """
        Get identified causal graph

        Returns
        -------
        CausalGraph object
        """
        causation = self.identifier_.causal_matrix_ \
            if isinstance(self.identifier_, IdentifierWithDiscovery) else None

        if causation is not None:
            threshold = causation.values.diagonal().max()
            m = BaseDiscovery.matrix2dict(causation, threshold=threshold)
        else:
            m = {}
            fmt = partial(utils.format, line_limit=1, line_width=20)
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

    def causal_effect(self, test_data=None, treatment=None, treat=None, control=None, target_outcome=None,
                      quantity='ATE', combine_treatment=False, return_detail=False,
                      **kwargs):
        """
        Estimate the causal effect.

        Parameters
        ----------
        test_data : pd.DataFrame, default None
            The test data to evaluate the causal effect.
            If None, the training data is used.
        treatment : str or list, optional
            Treatment names, should be subset of  attribute **treatment_**,
            default all elements in attribute **treatment_**
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
        target_outcome : outcome value, optional
            Only effective when the outcome is discrete. Default the last one in y_encoder_.classes_.
        quantity : str, optional, default 'ATE'
            'ATE' or 'ITE', default 'ATE'.
        combine_treatment : bool, default False
            Only supported for discrete treatment.
        return_detail: bool, default False
            If True, return effect details in result when quantity=='ATE'.
            Effective when quantity='ATE' only.
        kwargs : dict,
            Other options to call estimator.estimate().

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
        assert test_data is None or isinstance(test_data, pd.DataFrame)

        if self.discrete_treatment:
            fn = self._causal_effect_discrete
        else:
            fn = self._causal_effect_continuous

        if self.discrete_outcome and target_outcome is None:
            target_outcome = self.y_encoder_.classes_[-1]

        options = dict(treatment=treatment, treat=treat, control=control,
                       target_outcome=target_outcome,
                       quantity=quantity, combine_treatment=combine_treatment,
                       return_detail=return_detail)
        options.update(kwargs)
        return fn(test_data, **options)

    def _causal_effect_discrete(self, test_data=None, treatment=None, treat=None, control=None, target_outcome=None,
                                quantity='ATE', combine_treatment=False, return_detail=False,
                                **kwargs):
        # dfs = []
        # for i, x in enumerate(self.treatment_):
        #     est = self.estimators_[x]
        #     xe = self.x_encoders_[x]
        #     if control is not None:
        #         assert control[i] in xe.classes_.tolist(), f'Invalid {x} control "{control[i]}"'
        #         control_i = control[i]
        #     else:
        #         control_i = xe.classes_[0]
        #
        #     if treat is not None:
        #         assert treat[i] in xe.classes_.tolist(), f'Invalid {x} treat "{treat[i]}"'
        #         treats = [treat[i], ]
        #     else:
        #         if test_data is not None and x in test_data.columns.tolist():
        #             treats = np.unique(test_data[x]).tolist()
        #         else:
        #             treats = xe.classes_.tolist()
        #         treats = filter(lambda _: _ != control_i, treats)
        #
        #     if test_data is not None and x in test_data.columns.tolist():
        #         test_data[x] = xe.transform(test_data[x])
        #     c = xe.transform([control_i]).tolist()[0]
        #     for treat_i in treats:
        #         t = xe.transform([treat_i]).tolist()[0]
        #         effect = est.estimate(data=test_data, treat=t, control=c)
        #         if quantity == 'ATE':
        #             s = pd.Series(dict(mean=effect.mean(),
        #                                min=effect.min(),
        #                                max=effect.max(),
        #                                std=effect.std(),
        #                                ))
        #             if return_detail:
        #                 s['detail'] = effect.ravel()
        #         else:
        #             s = pd.Series(effect.ravel())
        #         s.name = (x, f'{treat_i} vs {control_i}')
        #         dfs.append(s)
        def _inverse_transform(x, t, c):
            if isinstance(x, (list, tuple)):
                assert isinstance(t, (list, tuple)) and isinstance(c, (list, tuple))
                assert len(x) == len(t) == len(c)
                tc = []
                for xi, ti, ci in zip(x, t, c):
                    tc.append(self.x_encoders_[xi].inverse_transform([ti, ci]).tolist())
                return np.array(tc).T.tolist()
            else:
                return self.x_encoders_[x].inverse_transform([t, c]).tolist()

        def _to_ate(effect, x, t, c, preprocessed_data):
            t, c = _inverse_transform(x, t, c)
            data = dict(mean=effect.mean(),
                        min=effect.min(),
                        max=effect.max(),
                        std=effect.std(),
                        )
            if return_detail:
                data['detail'] = effect.ravel()
            return pd.Series(data, name=(f'{x}', f'{t} vs {c}'))

        def _to_ite(effect, x, t, c, preprocessed_data):
            t, c = _inverse_transform(x, t, c)
            return pd.Series(effect.ravel(), name=(f'{x}', f'{t} vs {c}'))

        options = dict(treatment=treatment, treat=treat, control=control,
                       target_outcome=target_outcome, combined=combine_treatment)
        options.update(kwargs)

        if quantity == 'ATE':
            dfs = self._map_effect(_to_ate, test_data, **options)
        else:
            dfs = self._map_effect(_to_ite, test_data, **options)

        result = pd.concat(dfs, axis=1)
        if quantity == 'ATE':
            result = result.T
        return result

    def _causal_effect_continuous(self, test_data=None, treatment=None, treat=None, control=None, target_outcome=None,
                                  quantity='ATE', combine_treatment=False, return_detail=False,
                                  **kwargs):
        assert not combine_treatment, \
            '"combine_treatment" is not supported for continuous treatment.'

        if treatment is None:
            treatment = self.treatment_

        test_data_preprocessed = self._preprocess(test_data)
        treat = self._safe_treat_control(treatment, treat, 'treat')
        control = self._safe_treat_control(treatment, control, 'control')

        dfs = []
        for i, x in enumerate(treatment):
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
            effect = est.estimate(data=test_data_preprocessed, treat=treat_i, control=control_i, **kwargs)
            if self.fn_cost is not None:
                effect = utils.cost_effect(self.fn_cost, test_data, effect, self.effect_name)
            if quantity == 'ATE':
                s = pd.Series(dict(mean=effect.mean(),
                                   min=effect.min(),
                                   max=effect.max(),
                                   std=effect.std(),
                                   ))
                if return_detail:
                    s['detail'] = effect.ravel()
            else:
                s = pd.Series(effect.ravel())
            s.name = x
            dfs.append(s)
        result = pd.concat(dfs, axis=1)
        if quantity == 'ATE':
            result = result.T
        return result

    def individual_causal_effect(self, test_data, control=None, target_outcome=None):
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
        target_outcome : outcome value, optional
            Only effective when  the outcome is discrete. Default the last one in y_encoder_.classes_.

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
        assert all(t in test_data.columns.tolist() for t in self.treatment_)

        if self.discrete_treatment:
            return self._individual_causal_effect_discrete(test_data, control, target_outcome)
        else:
            return self._individual_causal_effect_continuous(test_data, control, target_outcome)

    def _individual_causal_effect_discrete(self, test_data, control=None, target_outcome=None):
        test_data_preprocessed = self._preprocess(test_data)
        control = self._safe_treat_control(self.treatment_, control, 'control')

        if control is None:
            control = [self.x_encoders_[x].classes_.tolist()[0] for x in self.treatment_]
        if target_outcome is not None:
            target_outcome = self._inverse_transform_outcome(target_outcome)

        dfs = []
        for ri in test_data_preprocessed.index:
            row_df = test_data_preprocessed.loc[ri:ri]
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
                    if self.discrete_outcome:
                        effect = est.estimate(data=row_df, treat=t, control=c, target_outcome=target_outcome)
                    else:
                        effect = est.estimate(data=row_df, treat=t, control=c)
                    effect = effect.ravel()[0]
                s = pd.Series(dict(effect=effect),
                              name=(ri, x, f'{treat_i} vs {control_i}'))
                dfs.append(s)
        return pd.concat(dfs, axis=1).T

    def _individual_causal_effect_continuous(self, test_data, control=None, target_outcome=None):
        test_data_preprocessed = self._preprocess(test_data)
        control = self._safe_treat_control(self.treatment_, control, 'control')

        dfs = []
        for i, x in enumerate(self.treatment_):
            est = self.estimators_[x]
            treat_i = test_data_preprocessed[x]
            control_i = control[i] if control is not None else None
            effect = est.estimate(data=test_data_preprocessed, treat=treat_i, control=control_i)
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
            If str, it should be one of the fitted attribute **treatment_**.
            If None, the first element in the attribute **treatment_** is used.
        Returns
        -------
        pd.Series
            The counterfactual prediction
        """
        assert not self.discrete_outcome, 'whatif only support continuous outcome'
        assert test_data is not None and new_value is not None
        assert treatment is None or isinstance(treatment, str)
        if isinstance(treatment, str):
            assert treatment in self.treatment_
        if treatment is None:
            treatment = self.treatment_[0]

        estimator = self.estimators_[treatment]
        if estimator.is_discrete_treatment:
            return self._whatif_discrete(test_data, new_value, treatment, estimator)
        else:
            return self._whatif_continuous(test_data, new_value, treatment, estimator)

    def _whatif_discrete(self, test_data, new_value, treatment, estimator):
        data = self._preprocess(test_data, encode_treatment=False)

        y_old = data[self.outcome_]
        old_value = data[treatment]
        xe = self.x_encoders_[treatment]

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
                t_encoded, c_encoded = xe.transform([t, c]).tolist()
                eff = estimator.estimate(data_rows, treat=t_encoded, control=c_encoded)
            if self.fn_cost is not None:
                eff = utils.cost_effect(self.fn_cost, test_data, eff, self.effect_name)
            effect.append(pd.DataFrame(dict(e=eff.ravel()), index=tc_rows.index))
        effect = pd.concat(effect, axis=0)
        assert len(effect) == len(df)
        df['e'] = effect['e']  # align indices

        y_new = y_old + df['e']
        return y_new

    def _whatif_continuous(self, test_data, new_value, treatment, estimator):
        data = self._preprocess(test_data)

        y_old = data[self.outcome_]
        old_value = data[treatment].values
        if isinstance(new_value, pd.Series):
            new_value = new_value.values
        effect = estimator.estimate(data, treat=new_value, control=old_value)
        if self.fn_cost is not None:
            effect = utils.cost_effect(self.fn_cost, test_data, effect, self.effect_name)
        y_new = y_old + effect.ravel()
        return y_new

    def score(self, test_data=None, treat=None, control=None, target_outcome=None, scorer='auto'):
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
        target_outcome : outcome value, optional
            Only effective when  the outcome is discrete. Default the last one in y_encoder_.classes_.
        scorer: str, default 'auto'
            One of 'auto', 'rloss', 'auuc', 'qini'. default 'rloss'.

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

    def _score_auuc_qini(self, test_data, treatment=None, treat=None, control=None, target_outcome=None, scorer='auuc'):
        # treatment = utils.to_list(treatment, 'treatment') if treatment is not None else self.treatment_
        # self._check_test_data('_score_auuc_qini', test_data,
        #                       treatment=treatment,
        #                       allow_data_none=False,
        #                       check_discrete_treatment=True,
        #                       check_treatment=True,
        #                       check_outcome=True)
        #
        # def scoring(effect, x, treat, control, preprocessed_data):
        #     df = pd.DataFrame(dict(effect=effect,
        #                            x=preprocessed_data[x],
        #                            y=preprocessed_data[self.outcome_],
        #                            ))
        #     if scorer == 'auuc':
        #         s = L.auuc_score(df, outcome='y', treatment='x', treat=treat, control=control, random_name=None)
        #     else:
        #         s = L.qini_score(df, outcome='y', treatment='x', treat=treat, control=control, random_name=None)
        #     return s['effect']
        #
        # sa = self._map_effect(scoring, test_data, treatment=treatment, treat=treat, control=control)
        # score = np.mean(sa)
        um = self.uplift_model(test_data, treatment=treatment, treat=treat, control=control,
                               target_outcome=target_outcome)
        if scorer == 'auuc':
            s = um.auuc_score()
        else:
            s = um.qini_score()
        assert isinstance(s, pd.Series)
        score = s.mean()
        return score

    def _score_rloss(self, test_data=None, treat=None, control=None):
        scorers = self._create_scorers(test_data, scorer='rloss')
        test_data = self._preprocess(test_data, encode_treatment=True)
        treat = self._safe_treat_control(self.treatment_, treat, 'treat')
        control = self._safe_treat_control(self.treatment_, control, 'control')
        fit_options = drop_none(adjustment=self.adjustment_, covariate=self.covariate_, instrument=self.instrument_)

        sa = []
        for i, (x, scorer) in enumerate(scorers.items()):
            logger.info(f'fit scorer for {x} with {scorer}')
            if self.discrete_treatment:
                xe = self.x_encoders_[x]
                treat_i = xe.transform([treat[i], ]).tolist()[0] if treat is not None else None
                control_i = xe.transform([control[i], ]).tolist()[0] if control is not None else None
            else:
                treat_i = treat[i] if treat is not None else None
                control_i = control[i] if control is not None else None
            scorer.fit(test_data, self.outcome_, x, **fit_options)
            estimator = self.estimators_[x]
            score_options = drop_none(treat=treat_i, control=control_i)
            sa.append(scorer.score(estimator, **score_options))
        score = np.mean(sa)

        return score

    def _effect_array(self, test_data, preprocessed_data, treatment, control=None, target_outcome=None):
        if treatment is None:
            treatment = self.treatment_[:2]
        else:
            treatment = utils.to_list(treatment, 'treatment')

        if len(treatment) > 2:
            raise ValueError(f'2 treatment are supported at most.')

        control = self._safe_treat_control(treatment, control, 'control')

        if self.discrete_treatment:
            estimator = self._get_estimator(treatment)
            if control is not None:
                control = [self.x_encoders_[x].transform([control[i]]).tolist()[0] for i, x in enumerate(treatment)]
                if len(treatment) == 1:
                    control = control[0]
            options = drop_none(control=control, )
            if target_outcome is not None \
                    and 'target_outcome' in inspect.signature(estimator.effect_nji).parameters.keys():
                options['target_outcome'] = self._inverse_transform_outcome(target_outcome)
            effect = estimator.effect_nji(preprocessed_data, **options)
            assert isinstance(effect, np.ndarray)
            assert effect.ndim == 3

            if self.discrete_outcome and effect.shape[1] > 1:
                assert effect.shape[1] == len(self.y_encoder_.classes_)
                if target_outcome is not None:
                    nz = np.nonzero(self.y_encoder_.classes_ == target_outcome)[0]
                    if len(nz) == 0:
                        assert ValueError(f'Invalid target_outcome: "{target_outcome}"')
                    effect = effect[:, nz[0]:nz[0] + 1, :]
                else:
                    effect = effect[:, -1:, :]

            assert effect.ndim == 3 and effect.shape[1] == 1
            effect_array = effect.squeeze(axis=1)
            classes = [self.x_encoders_[x].classes_.tolist() for x in treatment]
            effect_labels = list(map(lambda pair: '_'.join(map(str, pair)), product(*classes)))
        else:
            effects = []
            for i, x in enumerate(treatment):
                estimator = self.estimators_[x]
                ci = control[i] if control is not None else None
                options = drop_none(control=ci, target_outcome=target_outcome)
                effect = estimator.effect_nji(preprocessed_data, **options)
                assert isinstance(effect, np.ndarray)
                assert effect.ndim == 3 and effect.shape[1] == 1
                effects.append(effect.squeeze(axis=1))
            effect_array = np.hstack(effects)
            effect_labels = treatment

        assert effect_array.shape[1] == len(effect_labels)

        if self.fn_cost is not None:
            effect_array = utils.cost_effect_array(self.fn_cost, test_data, effect_array, self.effect_name)

        return effect_array, effect_labels

    def policy_interpreter(self, test_data, treatment=None, control=None, target_outcome=None, **kwargs):
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
        target_outcome : outcome value, optional
            Only effective when  the outcome is discrete. Default the last one in y_encoder_.classes_.
        kwargs : options to initialize the PolicyInterpreter

        Returns
        -------
        PolicyInterpreter :
            The fitted PolicyInterpreter object
        """
        # from ylearn.effect_interpreter.policy_interpreter import PolicyInterpreter
        from ._wrapper import WrappedPolicyInterpreter
        preprocessed_data = self._preprocess(test_data, encode_treatment=True)
        effect_array, labels = self._effect_array(test_data, preprocessed_data, treatment, control=control,
                                                  target_outcome=target_outcome)
        pi = WrappedPolicyInterpreter(**kwargs)
        pi.fit(preprocessed_data, covariate=self.covariate_, est_model=None,
               effect_array=effect_array, treatment_names=labels)
        pi.transformer = self.preprocessor_
        return pi

    def _check_test_data(self, reason, test_data,
                         treatment=None,
                         allow_data_none=True,
                         check_covariate=True,
                         check_instrument=False,
                         check_treatment=False,
                         check_outcome=False,
                         check_discrete_treatment=False,
                         ):

        if check_discrete_treatment:
            assert self.discrete_treatment, f'[{reason}] Only discrete treatment is supported.'

        if test_data is None:
            assert allow_data_none, f'[{reason}] test_data is required.'
            return

        columns = test_data.columns.tolist()

        if check_covariate:
            notfound = [c for c in self.covariate_ if c not in columns]
            assert len(notfound) == 0, f'[{reason}] Not found covariate: {notfound} in test_data.'

        if check_instrument:
            notfound = [c for c in self.instrument_ if c not in columns]
            assert len(notfound) == 0, f'[{reason}] Not found instrument: {notfound} in test_data.'

        if check_treatment:
            if treatment is None:
                treatment = self.treatment_
            notfound = [c for c in treatment if c not in columns]
            assert len(notfound) == 0, f'[{reason}] Not found treatment: {notfound} in test_data.'

        if check_outcome:
            assert self.outcome_ in columns, \
                f'[{reason}] Not found outcome {self.outcome_} in test_data.'

    def _map_effect(self, handler, test_data, treatment=None, treat=None, control=None,
                    target_outcome=None, combined=False):
        assert self.discrete_treatment, 'Only discrete treatment is supported'

        test_data_preprocessed = self._preprocess(test_data, encode_outcome=True, encode_treatment=True)
        treatment = utils.to_list(treatment, 'treatment') if treatment is not None else self.treatment_
        if combined:
            assert len(treatment) <= 2, f'2 treatment are supported at most.'

        treat = self._safe_treat_control(treatment, treat, 'treat')
        control = self._safe_treat_control(treatment, control, 'control')

        if combined:
            itr = self._permute_treat_control_combined(treatment, treat, control, encode=True)
        else:
            itr = self._permute_treat_control_sequential(treatment, treat, control, encode=True)

        if target_outcome is not None:
            # target_outcome = self.y_encoder_.inverse_transform([target_outcome]).tolist()[0]
            target_outcome = self.y_encoder_.transform([target_outcome]).tolist()[0]

        result = []
        for x, t, c in itr:
            est = self._get_estimator(x)
            if self.discrete_outcome:
                effect = est.estimate(data=test_data_preprocessed, treat=t, control=c, target_outcome=target_outcome)
            else:
                effect = est.estimate(data=test_data_preprocessed, treat=t, control=c)
            if self.fn_cost is not None and test_data is not None:
                effect = utils.cost_effect(self.fn_cost, test_data, effect, self.effect_name)
            result.append(handler(effect.ravel(), x, t, c, test_data_preprocessed))

        return result

    def _permute_treat_control_combined(self, treatment, treats, control, encode=False):
        assert self.discrete_treatment, 'Only discrete treatment is supported'

        if treatment is None:
            treatment = self.treatment_
        else:
            treatment = utils.to_list(treatment, 'treatment')
        assert len(treatment) > 0

        treats = self._safe_treat_control_list(treatment, treats, 'treats')
        control = self._safe_treat_control(treatment, control, 'control')

        def _encode(t_or_c):
            assert isinstance(t_or_c, (tuple, list)) and len(t_or_c) == len(treatment)
            t_or_c = [self.x_encoders_[x].transform([t_or_c[i]]).tolist()[0] for i, x in enumerate(treatment)]
            return t_or_c

        if control is None:
            control = [self.x_encoders_[x].classes_.tolist()[0] for x in treatment]

        if treats is None:
            if len(treatment) == 1:
                treats = [(t,) for t in self.x_encoders_[treatment[0]].classes_.tolist()]
            else:  # len(treatment)==2:
                classes = [self.x_encoders_[x].classes_.tolist() for x in treatment]
                treats = product(*classes)

        if encode:
            control = _encode(control)
            treats = map(_encode, treats)

        single_treatment = len(treatment) == 1
        for t in treats:
            if tuple(t) == tuple(control):
                continue
            if single_treatment:
                yield treatment[0], t[0], control[0]
            else:
                yield treatment, tuple(t), tuple(control)

    def _permute_treat_control_sequential(self, treatment, treats, control, encode=False):
        assert self.discrete_treatment, 'Only discrete treatment is supported'

        if treatment is None:
            treatment = self.treatment_
        else:
            treatment = utils.to_list(treatment, 'treatment')
        assert len(treatment) > 0
        # assert len(treatment) <= 2, f'2 treatment are supported at most.'

        treats = self._safe_treat_control_list(treatment, treats, 'treats')
        control = self._safe_treat_control(treatment, control, 'control')

        if control is None:
            control = [self.x_encoders_[x].classes_.tolist()[0] for x in treatment]
        for i, x in enumerate(treatment):
            xe = self.x_encoders_[x]

            assert control[i] in xe.classes_.tolist(), f'Invalid {x} control "{control[i]}"'
            control_i = control[i]

            if treats is not None:
                # assert treats[i] in xe.classes_.tolist(), f'Invalid {x} treat "{treats[i]}"'
                treats_i = [t[i] for t in treats]
            else:
                treats_i = filter(lambda _: _ != control_i, xe.classes_.tolist())

            for treat_i in treats_i:
                if encode:
                    t, c = xe.transform([treat_i, control_i]).tolist()
                    yield x, t, c
                else:
                    yield x, treat_i, control_i

    def uplift_model(self, test_data, treatment=None, treat=None, control=None, target_outcome=None, name=None,
                     random=None):
        """
        Get uplift model over one treatment.

        Parameters
        ----------
        test_data : pd.DataFrame, required
            The test data columns should contain all covariate, treatment and outcome.
        treatment : str, default None
            Treatment name.
            If str, it should be one of the fitted attribute **treatment_**.
            If None, the first element in the attribute **treatment_** is used.
        treat : treatment value, default None
            If None, the last element in the treatment encoder's attribute **classes_** is used.
        control : treatment value, default None
            If None, the first element in the treatment encoder's attribute **classes_** is used.
        target_outcome : outcome value, optional
            Only effective when  the outcome is discrete. Default the last one in y_encoder_.classes_.
        name : str, default None
            Lift name.
            If None, treat value is used.
        random : str, default none
            Lift name for random generated data.
            if None, no random lift is generated.
        -------
        pd.Series
            The counterfactual prediction
        """
        assert self.discrete_treatment, 'Uplift_model support discrete treatment only.'

        treatment = utils.to_list(treatment, 'treatment') if treatment is not None else self.treatment_
        self._check_test_data('uplift_model', test_data,
                              treatment=treatment,
                              allow_data_none=False,
                              check_discrete_treatment=True,
                              check_treatment=True,
                              check_outcome=True)

        test_data_preprocessed = self._preprocess(test_data, encode_treatment=True)
        treat = self._safe_treat_control(treatment, treat, 'treat')
        control = self._safe_treat_control(treatment, control, 'control')
        target_outcome = self._inverse_transform_outcome(target_outcome)

        encoders = [self.x_encoders_[x] for x in treatment]
        estimator = self._get_estimator(treatment)

        if treat is None:
            treat = [e.classes_.tolist()[-1] for e in encoders]
        if control is None:
            control = [e.classes_.tolist()[0] for e in encoders]
        if name is None:
            name = '_'.join(map(str, treat))

        t_encoded, c_encoded = utils.transform_treat_control(encoders, treat=treat, control=control)
        data_tc = utils.select_by_treat_control(test_data_preprocessed, treatment, t_encoded, c_encoded)

        if self.discrete_outcome:
            effect = estimator.estimate(data_tc, treat=t_encoded, control=c_encoded, target_outcome=target_outcome)
        else:
            effect = estimator.estimate(data_tc, treat=t_encoded, control=c_encoded)
        df_lift = pd.DataFrame({
            name: effect.ravel(),
            '_x_': utils.encode_treat_control(data_tc, treatment=treatment, treat=t_encoded, control=c_encoded),
            '_y_': data_tc[self.outcome_]
        }, index=data_tc.index)
        um = L.UpliftModel()
        um.fit(df_lift, treatment='_x_', outcome='_y_', random=random)
        return um

    def plot_policy_interpreter(self, test_data, treatment=None, control=None, target_outcome=None, **kwargs):
        """
        Plot the policy interpreter

        Returns
        -------
        PolicyInterpreter :
            The fitted PolicyInterpreter object
        """
        pi = self.policy_interpreter(test_data, treatment=treatment, control=control, target_outcome=target_outcome,
                                     **kwargs)
        pi.plot()
        return pi

    def plot_causal_graph(self):
        """
        Plot the causal graph.
        """
        import pydot

        values = dict(WLIST=self.adjustment_, VLIST=self.covariate_, XLIST=self.treatment_,
                      YLIST=self.outcome_, ZLIST=self.instrument_)
        if not utils.is_empty(self.instrument_) and not utils.is_empty(self.adjustment_):
            dot_string = GRAPH_STRING_WZ
        elif not utils.is_empty(self.instrument_):
            dot_string = GRAPH_STRING_Z
        elif not utils.is_empty(self.adjustment_):
            dot_string = GRAPH_STRING_W
        else:
            dot_string = GRAPH_STRING_BASE

        for k, v in values.items():
            if dot_string.find(k) >= 0:
                width = 40 if k == 'ZLIST' else 64
                dot_string = dot_string.replace(k, utils.format(v, line_width=width))
        graph = pydot.graph_from_dot_data(dot_string)[0]
        view_pydot(graph, prog='fdp')

    def __repr__(self):
        return to_repr(self)
