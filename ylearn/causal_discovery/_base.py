from collections import OrderedDict

import networkx as nx
import numpy as np
import pandas as pd

from ylearn.utils import to_repr


class BaseDiscovery:
    def __call__(self, data, *, return_dict=False, threshold=None, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def trim_cycle(matrix):
        assert isinstance(matrix, np.ndarray)
        assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]

        g = nx.from_numpy_matrix(matrix, create_using=nx.DiGraph)  # .reverse()

        def trim_with_weight():
            edge_weights = nx.get_edge_attributes(g, 'weight')
            edges = sorted(edge_weights.keys(), key=lambda e: edge_weights[e], reverse=True)
            for X, Y in edges:
                if nx.has_path(g, Y, X):
                    paths = list(nx.all_shortest_paths(g, Y, X, weight='weight'))
                    for p in paths:
                        es = sorted(zip(p[:-1], p[1:]), key=lambda e: edge_weights[e])
                        u, v = es[0]
                        if g.has_edge(u, v):
                            g.remove_edge(u, v)
                    return True
            return False

        while trim_with_weight():
            pass

        assert nx.is_directed_acyclic_graph(g)
        return nx.to_numpy_array(g)

    @staticmethod
    def matrix2dict(matrix, names=None, depth=None):
        assert isinstance(matrix, (np.ndarray, pd.DataFrame))
        assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]
        assert names is None or len(names) == matrix.shape[0]

        matrix = matrix.copy()
        n = matrix.shape[0]

        if names is None:
            if isinstance(matrix, pd.DataFrame):
                names = matrix.columns.tolist()
            else:
                names = range(n)

        if isinstance(matrix, pd.DataFrame):
            matrix = matrix.values

        g = nx.from_numpy_matrix(matrix, create_using=nx.DiGraph).reverse()

        d = OrderedDict()
        for i, name in enumerate(names):
            t = set(c[1] for c in nx.dfs_edges(g, i, depth_limit=depth))
            d[name] = [names[j] for j in range(n) if j in t]

        return d

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
            for col in range(row + 1, matrix.shape[1]):
                if hasattr(matrix, 'iloc'):
                    v = matrix.iloc[row, col]
                    vt = matrix.iloc[col, row]
                else:
                    v = matrix[row, col]
                    vt = matrix[row, col]
                if abs(v) >= abs(vt):
                    r.append([row, col, v])
                else:
                    r.append([col, row, vt])

        if names is not None:
            r = [[names[row[0]], names[row[1]], row[2]] for row in r]

        return r

    @staticmethod
    def matrix2df(matrix, names=None, ):
        r = BaseDiscovery.matrix2array(matrix, names=names)
        return pd.DataFrame(r, columns=['from', 'to', 'prob'])

    def __repr__(self):
        return to_repr(self)
