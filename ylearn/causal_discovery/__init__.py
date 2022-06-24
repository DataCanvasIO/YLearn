"""Causal discovery algorithms.
"""
from ._base import BaseDiscovery

try:
    from ._discovery import CausalDiscovery
except ImportError:
    pass

# from ._discovery import CausalDiscovery