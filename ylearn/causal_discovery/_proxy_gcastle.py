"""
A proxy to Huawei Noah's Ark Lab gCastle.
see: https://github.com/huawei-noah/trustworthyAI/tree/master/gcastle
"""
import copy
import os

import numpy as np
import pandas as pd

if os.getenv('CASTLE_BACKEND') is None:
    os.environ['CASTLE_BACKEND'] = 'pytorch'

from castle import algorithms as A
from castle.common import BaseLearner

from ylearn.utils import drop_none, set_random_state, logging
from ._base import BaseDiscovery

logger = logging.get_logger(__name__)


class GCastleProxy(BaseDiscovery):
    def __init__(self, learner=None, random_state=None, **kwargs):
        assert learner is None or isinstance(learner, (str, BaseLearner))
        if isinstance(learner, str):
            assert hasattr(A, learner), f'Not found learner "{learner}" from gcastle'
            c = getattr(A, learner)
            assert issubclass(c, BaseLearner)

        self.learner = learner
        self.options = kwargs.copy()
        self.random_state = random_state

    def _create_learner(self):
        if isinstance(self.learner, BaseLearner):
            return copy.copy(self.learner)
        else:
            c = getattr(A, self.learner) if self.learner is not None else A.PC
            return c(**self.options)

    def __call__(self, data, *, return_dict=False, threshold=None, **kwargs):
        assert isinstance(data, (np.ndarray, pd.DataFrame))

        set_random_state(self.random_state)

        if isinstance(data, pd.DataFrame):
            columns = data.columns.tolist()
            data = data.values
        else:
            columns = None

        learner = self._create_learner()
        logger.info(f'discovery causation with {type(learner).__name__}')
        learner.learn(data)
        matrix = learner.causal_matrix

        if columns is not None:
            matrix = pd.DataFrame(matrix, columns=columns, index=columns)

        if return_dict:
            result = self.matrix2dict(matrix, **drop_none(threshold=threshold))
        else:
            result = matrix

        return result
