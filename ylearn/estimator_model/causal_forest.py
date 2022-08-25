import numbers
import numpy as np

from abc import abstractmethod
from copy import deepcopy

from sklearn.utils import check_random_state

from . import CausalTree, DML4CATE, BaseEstModel
from ._forest._base_forest import BaseCausalForest


"""
Three different kinds of causal forest will be implemented, including
 1. A causal forest directly serving as an average of a bunch of causal trees (honest or not)
 2. A causal forest by growing generalized random forest tree (these trees may grow in a dfferent way when compared to the causal tree)
 3. A causal forest by applying the local centering technique.
"""


# overall causal forest
class CausalForest(BaseCausalForest):
    def __init__(
        self,
    ):
        pass


# for iv
class IVCausalForest:
    def __init__(self) -> None:
        pass
