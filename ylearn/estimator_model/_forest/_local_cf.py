import numbers
import numpy as np

from copy import deepcopy

from sklearn.utils import check_random_state

from . import CausalTree, DML4CATE, BaseEstModel
from ._base_forest import BaseCausalForest

# causal forest with local centering technique
class LocalCausalForest(BaseCausalForest, DML4CATE):
    def __init__(
        self,
    ):
        pass
