import inspect
import sys as _sys
import time
import traceback as _traceback
from collections import defaultdict
from functools import partial

from ylearn.utils import logging

_LOG_LEVEL = 'info'
_VALUE_LEN_LIMIT = 10
_COLLECTION_LIMIT = 5

_TIC_TOC_TAG = 'tic_toc_'
_TIC_TOC_NAME_PREFIX = '[tic-toc]'
_TIC_TOC_NAME_SUFFIX = '@'

_stat_call_counter = defaultdict(int)
_stat_second_counter = defaultdict(int)


def tic_toc(log_level=_LOG_LEVEL, name=None, details=True):
    log_level = logging.to_level(log_level)

    return partial(_tic_toc_decorate, log_level, name, details)


def tit(fn, log_level=_LOG_LEVEL, name=None, details=False):
    log_level = logging.to_level(log_level)
    return _tic_toc_decorate(log_level, name, details, fn)


def _tic_toc_decorate(log_level, name, details, fn):
    assert callable(fn)

    if log_level < logging.get_level():
        return fn

    if hasattr(fn, _TIC_TOC_TAG):
        return fn

    if name is None:
        try:
            logger_name = f'{_TIC_TOC_NAME_PREFIX}{fn.__module__}.{fn.__qualname__}{_TIC_TOC_NAME_SUFFIX}'
        except:
            logger_name = f'{_TIC_TOC_NAME_PREFIX}{fn.__name__}{_TIC_TOC_NAME_SUFFIX}'
    else:
        logger_name = name

    logger = logging.get_logger(logger_name)
    logger.findCaller = _logger_find_caller

    fn_sig = inspect.signature(fn) if details else None

    def tic_toc_call(*args, **kwargs):
        tic = time.time()
        try:
            r = fn(*args, **kwargs)
            return r
        finally:
            toc = time.time()
            elapsed = toc - tic

            msg = f'elapsed {elapsed:.3f} seconds'
            if details and (len(args) > 0 or len(kwargs) > 0):
                ba = fn_sig.bind(*args, **kwargs)
                args = [f'<{k}>' if k == 'self' else f'{k}={_format_value(v)}'
                        for k, v in ba.arguments.items()]
                msg += f', details:\t{", ".join(args)}'

            logger.log(log_level, msg)

            _stat_call_counter[logger_name] += 1
            _stat_second_counter[logger_name] += elapsed

    setattr(tic_toc_call, _TIC_TOC_TAG, fn)

    return tic_toc_call


def _format_value(v, expand_collection=True):
    if v is None or isinstance(v, (int, float, bool, complex)):
        r = v
    elif isinstance(v, str):
        if len(v) > _VALUE_LEN_LIMIT:
            r = v[:_VALUE_LEN_LIMIT]
            r = f'{r}...[len={len(v)}]'
        else:
            r = v
    elif isinstance(v, (bytes, bytearray)):
        r = f'{type(v).__name__}[{len(v)}]'
    elif isinstance(v, type):
        r = v.__name__
    elif hasattr(v, 'shape'):
        r = f'{type(v).__name__}[shape={getattr(v, "shape")}]'
    elif hasattr(v, '__name__'):
        r = f'{type(v).__name__}[name={getattr(v, "__name__")}]'
    elif isinstance(v, dict) and expand_collection:
        r = type(v)()
        for k, v in v.items():
            r[_format_value(k, False)] = _format_value(v, False)
            if len(r) >= _COLLECTION_LIMIT:
                r['...'] = '...'
                break
    elif hasattr(v, '__iter__') and expand_collection:
        r = []
        for e in v:
            r.append(_format_value(e, False))
            if len(r) >= _COLLECTION_LIMIT:
                r.append('...')
                break
        r = type(v).__name__, r
    elif hasattr(v, '__len__'):
        r = f'{type(v).__name__}[len={len(v)}]'
    else:
        r = f'{type(v).__name__}'

    return r


class TicTocCfg:
    enabled = True
    auto_load = True

    level = 'info'
    details = True

    modules = {}


def load_cfg(cfg: TicTocCfg):
    log_level = logging.to_level(cfg.level)
    details = cfg.details

    for cls, v in cfg.modules.items():
        assert isinstance(v, (str, tuple, list))
        if isinstance(v, str):
            v = v.split(',')
        fns = filter(lambda x: isinstance(x, str) and len(x) > 0, v)

        for f in fns:
            assert isinstance(f, str)
            fn = getattr(cls, f, None)
            if fn is None:
                print(f'[tic-toc] Not found "{f}" in {cls.__name__}', file=_sys.stderr)
            elif not callable(fn):
                print(f'[tic-toc] Not callable: {cls.__name__}.{f}', file=_sys.stderr)
            else:
                fn_decorated = _tic_toc_decorate(log_level=log_level, name=None, details=details, fn=fn)
                setattr(cls, f, fn_decorated)


def report():
    r = {}
    for k, count in _stat_call_counter.items():
        seconds = _stat_second_counter[k]
        # name = k.lstrip(_TIC_TOC_NAME_PREFIX).rstrip(_TIC_TOC_NAME_SUFFIX)
        name = k[len(_TIC_TOC_NAME_PREFIX):-len(_TIC_TOC_NAME_SUFFIX)]
        r[name] = (count, seconds, seconds / count)

    return r


def report_as_dataframe():
    import pandas as pd
    names = []
    counts = []
    total_seconds = []
    average_seconds = []
    for name, v in report().items():
        names.append(name)
        counts.append(v[0])
        total_seconds.append(v[1])
        average_seconds.append(v[2])

    df = pd.DataFrame({
        'name': names,
        'count': counts,
        'total_second': total_seconds,
        'average_second': average_seconds
    })
    df.set_index('name', drop=True, inplace=True)
    df.sort_index(inplace=True)
    return df


###############################################################
# adapted from logging


# The definition of `findCaller` changed in Python 3.2,
# and further changed in Python 3.8
if _sys.version_info.major >= 3 and _sys.version_info.minor >= 8:

    def _logger_find_caller(stack_info=False, stacklevel=1):  # pylint: disable=g-wrong-blank-lines
        code, frame = logging._get_caller(6)
        sinfo = None
        if stack_info:
            sinfo = '\n'.join(_traceback.format_stack())
        if code:
            return (code.co_filename, frame.f_lineno, code.co_name, sinfo)
        else:
            return '(unknown file)', 0, '(unknown function)', sinfo
elif _sys.version_info.major >= 3 and _sys.version_info.minor >= 2:

    def _logger_find_caller(stack_info=False):  # pylint: disable=g-wrong-blank-lines
        code, frame = logging._get_caller(5)
        sinfo = None
        if stack_info:
            sinfo = '\n'.join(_traceback.format_stack())
        if code:
            return (code.co_filename, frame.f_lineno, code.co_name, sinfo)
        else:
            return '(unknown file)', 0, '(unknown function)', sinfo
else:
    def _logger_find_caller():  # pylint: disable=g-wrong-blank-lines
        code, frame = logging._get_caller(6)
        if code:
            return (code.co_filename, frame.f_lineno, code.co_name)
        else:
            return '(unknown file)', 0, '(unknown function)'

###############################################################

if TicTocCfg.enabled and TicTocCfg.auto_load:
    load_cfg(TicTocCfg)
