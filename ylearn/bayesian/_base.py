import copy
from collections import OrderedDict

import numpy as np
import pandas as pd

from pyro.poutine import Trace
from ylearn.utils import to_repr


class BObject(object):
    def __repr__(self):
        return to_repr(self)


class BState(BObject):
    """
    Bayesian network node state.
    """

    def encode(self, value):
        """
        expression value to trainable value
        """
        raise NotImplementedError()

    def decode(self, value):
        """
        trainable value to expression value
        """
        raise NotImplementedError()


class NumericalNodeState(BState):
    def __init__(self, mean, scale, min, max):
        self.mean = mean
        self.scale = scale
        self.min = min
        self.max = max

    def encode(self, value):
        return value

    def decode(self, value):
        return value

    encode.__doc__ = BState.encode.__doc__
    decode.__doc__ = BState.decode.__doc__


class CategoryNodeState(BState):
    def __init__(self, classes):
        assert len(classes) >= 2
        self.classes = np.sort(classes)

    @property
    def n_classes(self):
        return len(self.classes)

    @property
    def is_binary(self):
        return len(self.classes) == 2

    def encode(self, value):
        """
        classes to indices
        """
        r = np.searchsorted(self.classes, value)
        return r

    def decode(self, value):
        """
        indices to classes
        """
        r = np.take(self.classes, value)
        return r


class BCollector(BObject):
    """
    Sample collector
    """

    def __init__(self, name, state):
        self.name = name
        self.state = copy.copy(state)

    def __call__(self, value):
        raise NotImplementedError()

    def to(self, result):
        raise NotImplementedError()

    def to_df(self):
        result = OrderedDict()
        self.to(result)
        return pd.DataFrame(result)


class BCollectorList(BObject):
    """
    Sample collector list
    """

    def __init__(self, names, collector_creator):
        assert isinstance(names, (list, tuple))
        assert callable(collector_creator) or isinstance(collector_creator, (tuple, list))

        if isinstance(collector_creator, (tuple, list)):
            assert len(names) == len(collector_creator)
        else:
            collector_creator = list(map(collector_creator, names))

        self.names = names
        self.collectors = collector_creator

    def __call__(self, value):
        if isinstance(value, Trace):
            for name, collector in zip(self.names, self.collectors):
                collector(value.nodes[name]['value'])
        elif isinstance(value, dict):
            for name, collector in zip(self.names, self.collectors):
                collector(value[name])
        else:
            raise ValueError(f'Unsupported value type: "{type(value).__name__}"')

    def to(self, result):
        for collector in self.collectors:
            collector.to(result)

    def to_df(self):
        result = OrderedDict()
        self.to(result)
        return pd.DataFrame(result)
