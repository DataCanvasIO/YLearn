"""Causal discovery algorithms.
"""
from ._base import BaseDiscovery

try:
    from ._discovery import DagDiscovery
except ImportError:
    pass
