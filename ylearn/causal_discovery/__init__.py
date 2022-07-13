"""Causal discovery algorithms.
"""
from ._base import BaseDiscovery

try:
    from ._discovery import CausalDiscovery
except ImportError as e:
    _msg = f'{e}, install pytorch and try again.'


    class CausalDiscovery(BaseDiscovery):
        def __init__(self, *args, **kwargs):
            raise ImportError(_msg)

        def __call__(self, data, *, return_dict=False, threshold=None, **kwargs):
            raise ImportError(_msg)
