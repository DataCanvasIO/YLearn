"""
A proxy to pgmpy structure estimators
see: https://github.com/pgmpy/pgmpy
"""
import inspect

import networkx as nx
import numpy as np
import pandas as pd
from pgmpy import estimators as A

from ylearn.utils import set_random_state, logging
from ._base import BaseDiscovery

logger = logging.get_logger(__name__)

_default_options = dict(
    PC=dict(variant="stable",
            ci_test="pearsonr",  # default continuous datasets.
            show_progress=False),
)


class PgmProxy(BaseDiscovery):
    def __init__(self, learner='PC', random_state=None, **kwargs):
        assert isinstance(learner, str) and hasattr(A, learner), \
            f'Not found learner "{learner}" from pgmpy.estimators'
        c = getattr(A, learner)
        assert issubclass(c, A.StructureEstimator)

        self.learner = learner
        self.options = kwargs.copy()
        self.random_state = random_state

    def _create_learner(self, data, options):
        c = getattr(A, self.learner) if self.learner is not None else A.PC

        kwargs = {}
        for k in inspect.signature(c.__init__).parameters.keys():
            if k in options.keys():
                kwargs[k] = options.pop(k)
        return c(data, **kwargs)

    def __call__(self, data, *, return_dict=False, threshold=None, **kwargs):
        assert isinstance(data, (np.ndarray, pd.DataFrame))

        set_random_state(self.random_state)

        if isinstance(data, pd.DataFrame):
            df = data
        else:
            df = pd.DataFrame(data)

        options = _default_options.get(self.learner, {}).copy()
        options.update(**self.options)
        learner = self._create_learner(df, options)

        logger.info(f'discovery causation with {type(learner).__name__}')
        if isinstance(learner, A.PC):
            options['return_type'] = 'dag'
        dag = learner.estimate(**options)

        columns = df.columns.tolist()
        nodes = list(dag.nodes)
        assert set(nodes).issubset(set(columns))

        matrix_learned = pd.DataFrame(nx.to_numpy_array(dag, nodelist=nodes, weight=None),
                                      columns=nodes, index=nodes)
        matrix_full = pd.DataFrame(np.zeros((df.shape[1], df.shape[1])),
                                   columns=columns, index=columns)
        matrix_full = (matrix_full + matrix_learned).fillna(0.0)

        if isinstance(data, pd.DataFrame):
            matrix = matrix_full
        else:
            matrix = matrix_full.values

        if return_dict:
            result = self.matrix2dict(matrix)
        else:
            result = matrix

        return result
