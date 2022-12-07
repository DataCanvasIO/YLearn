"""Causal discovery algorithms.
"""
from ._base import BaseDiscovery

try:
    from ._discovery import CausalDiscovery
except ImportError as e:
    _msg_cd = f'{e}, install pytorch and try again.'


    class CausalDiscovery(BaseDiscovery):
        def __init__(self, *args, **kwargs):
            raise ImportError(_msg_cd)

try:
    from ._proxy_gcastle import GCastleProxy
except ImportError as e:
    _msg_gcastle = f'{e}, install gcastle and try again.'


    class GCastleProxy(BaseDiscovery):
        def __init__(self, *args, **kwargs):
            raise ImportError(_msg_gcastle)

try:
    from ._proxy_pgm import PgmProxy
except ImportError as e:
    _msg_pgm = f'{e}, install pgmpy and try again.'


    class PgmProxy(BaseDiscovery):
        def __init__(self, *args, **kwargs):
            raise ImportError(_msg_pgm)
