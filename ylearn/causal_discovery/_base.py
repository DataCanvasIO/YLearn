from collections import OrderedDict

import numpy as np
import pandas as pd

from ylearn.utils import to_repr


class BaseDiscovery:
    def __call__(self, data, *, return_dict=False, threshold=None, **kwargs):
        raise NotImplementedError()

    # @staticmethod
    # def _matrix2dict_(matrix, names=None, threshold=0.1):
    #     assert isinstance(matrix, np.ndarray)
    #     assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]
    #     assert feature_names is None or len(feature_names) == matrix.shape[0]
    #
    #     if names is None:
    #         names = [f'X{i}' for i in range(matrix.shape[0])]
    #
    #     df = pd.DataFrame(matrix, columns=names, index=names)
    #     df = df.T.abs()
    #
    #     m = OrderedDict()
    #     for f in names:
    #         m[f] = df[df[f] > threshold].index.tolist()
    #
    #     return m

    @staticmethod
    def matrix2dict(matrix, threshold=0.1, names=None, ):
        assert isinstance(matrix, (np.ndarray, pd.DataFrame))
        assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]
        assert names is None or len(names) == matrix.shape[0]

        if names is None and isinstance(matrix, pd.DataFrame):
            names = matrix.columns.tolist()

        r = OrderedDict()
        idx = np.arange(matrix.shape[0])
        for i in range(matrix.shape[0]):
            row = matrix.iloc[i] if hasattr(matrix, 'iloc') else matrix[i]
            to = np.where(np.where(np.abs(row) > threshold, idx, -1) >= 0)[0]
            if names is None:
                r[i] = to.tolist()
            else:
                r[names[i]] = [names[t] for t in to]

        return r

    @staticmethod
    def matrix2array(matrix, names=None, ):
        """
        Returns
        -------
            table of:
                from,to,prob
                ...
        """
        assert isinstance(matrix, (np.ndarray, pd.DataFrame))
        assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]
        assert names is None or len(names) == matrix.shape[0]

        if names is None and isinstance(matrix, pd.DataFrame):
            names = matrix.columns.tolist()

        r = []
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                v = matrix.iloc[row, col] if hasattr(matrix, 'iloc') else matrix[row, col]
                r.append([row, col, v])

        if names is not None:
            r = [[names[row[0]], names[row[1]], row[2]] for row in r]

        return r

    @staticmethod
    def matrix2df(matrix, names=None, ):
        r = BaseDiscovery.matrix2array(matrix, names=names)
        return pd.DataFrame(r, columns=['from', 'to', 'prob'])

    def __repr__(self):
        return to_repr(self)
