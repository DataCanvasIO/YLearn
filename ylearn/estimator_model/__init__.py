# from . import base_models, double_ml, doubly_robust, meta_learner,\
#     propensity_score

from ._factory import ESTIMATOR_FACTORIES
from ._permuted import PermutedDoublyRobust
from ._permuted import PermutedSLearner, PermutedTLearner, PermutedXLearner
from .approximation_bound import ApproxBound
from .base_models import BaseEstModel
from .double_ml import DoubleML
from .doubly_robust import DoublyRobust
from .effect_score import RLoss, PredLoss
from .ensemble import EnsembleEstModels
from .iv import NP2SLS
from .meta_learner import SLearner, TLearner, XLearner
from .propensity_score import InversePbWeighting, PropensityScore

try:
    from .deepiv import DeepIV
except ImportError as e:  # torch not ready
    _msg_deep_iv = f"{e}"

    class DeepIV:
        def __init__(self, *args, **kwargs):
            raise ImportError(_msg_deep_iv)

try:
    from .causal_tree import CausalTree
except ImportError as e:  # cython extension not ready
    _msg_causal_tree = f"{e}"

    class CausalTree:
        def __init__(self, *args, **kwargs):
            raise ImportError(_msg_causal_tree)

try:
    from ._generalized_forest._grf import GRForest
except ImportError as e:  # cython extension not ready
    _msg_grf = f"{e}"

    class GRForest:
        def __init__(self, *args, **kwargs):
            raise ImportError(_msg_grf)
try:
    from .causal_forest import CausalForest
except ImportError as e:  # cython extension not ready
    _msg_causal_forest = f"{e}"

    class CausalForest:
        def __init__(self, *args, **kwargs):
            raise ImportError(_msg_causal_forest)
