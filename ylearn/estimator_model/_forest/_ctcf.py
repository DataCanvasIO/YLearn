import numbers
import numpy as np

from abc import abstractmethod
from copy import deepcopy

from sklearn.utils import check_random_state

from . import CausalTree, DML4CATE, BaseEstModel
from ._base_forest import BaseCausalForest

# causal forest built with causal tree
class CausalTreeCausalForest:
    def __init__(self) -> None:
        pass
