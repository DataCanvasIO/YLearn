import pandas as pd
import torch

from . import _base


class DataLoader(_base.BObject):
    def __init__(self, state=None, data=None):
        assert state is not None or data is not None
        assert state is None or isinstance(state, dict)
        assert data is None or isinstance(data, pd.DataFrame)

        # self.data = data
        self.state = DataLoader.state_of(data) if state is None else state

    def spread(self, data):
        df = data.copy()
        result = {}
        for c in df.columns.tolist():
            v = self.state[c].encode(df[c].values)
            result[c] = torch.tensor(v)
        #
        # graph = self.graph
        # for node in graph.nodes:
        #     parents = graph.get_parents(node)
        #     if parents:
        #         try:
        #             result[f'{node}_inputs'] = torch.tensor(df[parents].values)
        #         except:
        #             pass

        return result

    @staticmethod
    def state_of(data: pd.DataFrame):
        state = {}
        for n in data.columns.tolist():
            c = data[n]
            if c.dtype.kind == 'f':
                state[n] = _base.NumericalNodeState(c.mean(), c.std(), c.min(), c.max())
            else:
                state[n] = _base.CategoryNodeState(c.unique())

        return state
