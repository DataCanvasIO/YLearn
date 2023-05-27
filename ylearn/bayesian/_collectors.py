import torch

from . import _base


class NumberSampleCollector(_base.BCollector):
    def __init__(self, name, state):
        assert isinstance(state, _base.NumericalNodeState)
        super().__init__(name, state)

        self.value = None
        self.n_sample = 0

    def __call__(self, value):
        if self.value is None:
            self.value = value
        else:
            self.value += value
        self.n_sample += 1

    def to(self, result):
        result[self.name] = self.value / self.n_sample


class CategorySampleCollector(_base.BCollector):
    def __init__(self, name, state, proba=False):
        assert isinstance(state, _base.CategoryNodeState)
        super().__init__(name, state)

        self.proba = proba
        self.ones = torch.eye(state.n_classes, dtype=torch.int)
        self.value = None
        self.n_sample = 0

    def __call__(self, value):
        value = self.ones.index_select(0, value)  # one-hot
        if self.value is None:
            self.value = value
        else:
            self.value += value
        self.n_sample += 1

    def to(self, result):
        if self.proba:
            value = self.value / self.n_sample
            for i, c in enumerate(self.state.classes):
                result[f'{self.name}_{c}'] = value[:, i]
        else:
            value = self.state.decode(torch.argmax(self.value, dim=1))
            result[self.name] = value
