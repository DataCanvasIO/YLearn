import numbers
import numpy as np

from abc import abstractmethod
from copy import deepcopy

from sklearn.utils import check_random_state

from . import CausalTree, DoubleML, BaseEstModel
from ._generalized_forest._base_forest import BaseCausalForest


"""
Two different kinds of causal forest will be implemented, including
 1. A causal forest directly serving as an average of a bunch of causal trees (honest or not)
 3. A causal forest by growing generalized random forest tree (these trees may grow in a dfferent
    way when compared to the causal tree) combined with the local centering technique.
"""


class CausalForest(BaseCausalForest):
    def __init__(
        self,
    ):
        pass


# for iv
class IVCausalForest:
    def __init__(self) -> None:
        pass
