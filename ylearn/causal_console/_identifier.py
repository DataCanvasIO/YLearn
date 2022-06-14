import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ylearn import sklearn_ex as skex
from ylearn.causal_discovery import DagDiscovery
from ylearn.causal_model import CausalModel, CausalGraph
from ylearn.utils import logging

logger = logging.get_logger(__name__)


def _is_number(dtype):
    return dtype.kind in {'i', 'f'}


class Identifier:
    def __init__(self, task):
        self.task = task

    def identify_treatment(self, data, outcome, count_limit, excludes=None):
        raise NotImplemented()

    def identify_aci(self, data, outcome, treatment):
        raise NotImplemented()


class DefaultIdentifier(Identifier):
    def identify_treatment(self, data, outcome, count_limit, excludes=None):
        X = data.copy()
        y = X.pop(outcome)

        if excludes is not None and len(excludes) > 0:
            X = X[[c for c in X.columns.tolist() if c not in excludes]]

        tf = skex.FeatureImportancesSelectionTransformer(
            task=self.task, strategy='number', number=count_limit, data_clean=False)
        tf.fit(X, y)
        treatment = tf.selected_features_

        return treatment

    def identify_aci(self, data, outcome, treatment):
        adjustment, instrument = None, None

        covariate = [c for c in data.columns.tolist() if c != outcome and c not in treatment]

        return adjustment, covariate, instrument


class IdentifierWithDiscovery(Identifier):
    def __init__(self, task, random_state, **kwargs):
        super().__init__(task)

        self.random_state = random_state
        self.discovery_options = kwargs.copy()
        self.causation_matrix_ = None

    def _discovery(self, data, outcome):
        logger.info('discovery causation')

        X = data.copy()
        y = X.pop(outcome)

        if not _is_number(y.dtype):
            y = LabelEncoder().fit_transform(y)

        preprocessor = skex.general_preprocessor()
        X = preprocessor.fit_transform(X, y)
        X[outcome] = y

        options = dict(random_state=self.random_state)
        if self.discovery_options is not None:
            options.update(self.discovery_options)

        dd = DagDiscovery(**options)
        return dd(data)

    def identify_treatment(self, data, outcome, count_limit, excludes=None):
        causation = self._discovery(data, outcome)
        assert isinstance(causation, pd.DataFrame) and outcome in causation.columns.tolist()

        treatment = causation[outcome].abs().sort_values(ascending=False)
        treatment = [i for i in treatment.index if treatment[i] > 0]
        if excludes is not None:
            treatment = [t for t in treatment if t not in excludes]

        if len(treatment) > count_limit:
            treatment = treatment[:count_limit]

        self.causation_matrix_ = causation

        return treatment

    def identify_aci(self, data, outcome, treatment):
        adjustment, covariate = None, None

        if self.causation_matrix_ is None:
            self.causation_matrix_ = self._discovery(data, outcome)
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
            instrument = []

        if len(instrument) == 0:
            ids = cm.identify(treatment, outcome, identify_method=('backdoor', 'simple'))
            covariate = list(set(ids['backdoor'][0]))
        else:
            covariate = [c for c in data.columns.tolist()
                         if c != outcome and c not in treatment and c not in instrument]

        return adjustment, covariate, instrument
