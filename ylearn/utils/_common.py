import contextlib as _contextlib
import inspect
import warnings
from collections import OrderedDict
from functools import partial

import numpy as np
import pandas as pd


class const:
    TASK_AUTO = 'auto'
    TASK_BINARY = 'binary'
    TASK_MULTICLASS = 'multiclass'
    TASK_REGRESSION = 'regression'
    TASK_MULTILABEL = 'multilabel'


def set_random_state(random_state):
    if random_state is not None:
        seed = random_state if isinstance(random_state, int) \
            else random_state.randint(0, 65535)
        np.random.seed(seed)

        try:
            import torch
            torch.random.manual_seed(seed)
        except:
            pass


def check_cols(data, *x, ):
    x = filter(None, x)
    all_cols = data.columns

    for iter_ in x:
        if isinstance(iter_, str):
            iter_ = {iter_}

        for i in iter_:
            assert i in all_cols, f'Nonexistent variable {i}.'


def to_df(**data):
    dfs = []
    for k, v in data.items():
        if v is None:
            pass  # ignore the item
        elif len(v.shape) == 1:
            dfs.append(pd.Series(v, name=k))
        elif v.shape[1] == 1:
            dfs.append(pd.DataFrame(v, columns=[k]))
        else:
            dfs.append(pd.DataFrame(
                v, columns=[f'{k}_{i}' for i in range(v.shape[1])]))
    df = pd.concat(dfs, axis=1)
    return df


def to_list(v, name=None):
    if v is None or isinstance(v, (list, tuple)):
        pass
    elif isinstance(v, str):
        v = [s.strip() for s in v.split(',')]
        v = [s for s in v if len(s) > 0]
    else:
        tag = name if name is not None else 'value'
        raise ValueError(f'Unexpected {tag}: {v}')

    return v


def join_list(*args):
    r = []
    for a in args:
        if a is None:
            pass
        elif isinstance(a, list):
            r += a
        elif isinstance(a, tuple):
            r += list(a)
        else:
            r += to_list(a)
    return r


@_contextlib.contextmanager
def context(msg):
    try:
        yield
    except Exception as ex:
        if ex.args:
            msg = u'{}: {}'.format(msg, ex.args[0])
        else:
            msg = str(msg)
        ex.args = (msg,) + ex.args[1:]
        raise


def is_notebook():
    """
    code from https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except:
        return False


def view_pydot(pdot_graph, prog='dot'):
    try:
        from IPython.display import Image, display

        img = Image(pdot_graph.create_png(prog=prog))
        display(img)
    except Exception as e:
        warnings.warn(f'Failed to display pydot image: {e}.')


def unique(y):
    if hasattr(y, 'unique'):
        uniques = set(y.unique())
    else:
        uniques = set(y)
    return uniques


def infer_task_type(y, *, dropna=True, excludes=None, regression_exponent=0.382):
    assert excludes is None or isinstance(excludes, (list, tuple, set))

    if len(y.shape) > 1 and y.shape[-1] > 1:
        labels = list(range(y.shape[-1]))
        task = const.TASK_MULTILABEL
        return task, labels

    uniques = unique(y)
    if dropna:
        if uniques.__contains__(np.nan):
            uniques.remove(np.nan)
        if uniques.__contains__(None):
            uniques.remove(None)
    if excludes is not None and len(excludes) > 0:
        uniques -= set(excludes)
    n_unique = len(uniques)
    labels = []

    if n_unique == 0:
        raise ValueError('Could not infer task type from empty "y"')
    elif n_unique == 1:
        raise ValueError(f'Could not infer task type from unique "{uniques}"')
    elif n_unique == 2:
        task = const.TASK_BINARY
    elif y.dtype.kind == 'f':
        task = const.TASK_REGRESSION
    elif y.dtype.kind == 'i':
        n_sample = len(y)
        if n_unique > n_sample ** regression_exponent:
            task = const.TASK_REGRESSION
        else:
            task = const.TASK_MULTICLASS
    else:
        task = const.TASK_MULTICLASS

    if task != const.TASK_REGRESSION:
        labels = pd.Series(list(uniques)).sort_values().tolist()

    return task, labels


def get_params(obj, include_default=False):
    def _get_sk_estimator_params(est):
        try:
            from sklearn.base import BaseEstimator
            if isinstance(est, BaseEstimator):
                return est.get_params()
            else:
                return None
        except ImportError:
            return None

    def _get_init_params(cls):
        init = cls.__init__
        if init is object.__init__:
            return []

        init_signature = inspect.signature(init)
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self']  # and p.kind != p.VAR_KEYWORD]
        return parameters

    sk_params = _get_sk_estimator_params(obj)
    if sk_params is not None:
        return sk_params

    out = OrderedDict()
    for p in _get_init_params(type(obj)):
        name = p.name
        value = getattr(obj, name, None)
        if include_default or value is not p.default:
            out[name] = value

    return out


def to_repr(obj, excludes=None):
    try:
        if excludes is None:
            excludes = []
        out = ['%s=%r' % (k, v) for k, v in get_params(
            obj).items() if k not in excludes]
        repr_ = ', '.join(out)
        return f'{type(obj).__name__}({repr_})'
    except Exception as e:
        return f'{type(e).__name__}:{e}, at <to_repr>: {type(obj).__name__}'


def to_snake_case(camel_str):
    last_isupper = None
    a = []
    for c in camel_str:
        u = c.isupper()
        if u:
            if last_isupper is False:
                a.append('_')
            a.append(c.lower())
        else:
            a.append(c)
        last_isupper = u
    s = "".join(a)
    while s.startswith("_"):
        s = s[1:]
    return s


def to_camel_case(snake_str):
    components = snake_str.split('_')
    return ''.join(x.title() for x in components)


def drop_none(**kwargs):
    r = {k: v for k, v in kwargs.items() if v is not None}
    return r


def convert2array(data, *S):
    assert isinstance(data, pd.DataFrame)

    def _get_array(cols):
        if cols is not None:
            r = data[cols].values
            if len(r.shape) == 1:
                r = np.expand_dims(r, axis=1)
        else:
            r = None
        return r

    S = map(_get_array, S)

    return tuple(S)


def check_fitted(fn, attr_name='_is_fitted', check=None, msg=None):
    assert callable(fn)

    sig = inspect.signature(fn)
    assert 'self' in sig.parameters.keys()

    def check_and_call(obj, *args, **kwargs):
        if callable(check):
            fitted = check(obj)
        else:
            fitted_tag = getattr(obj, attr_name, None)
            if check is not None:
                fitted = not (fitted_tag is check)
            else:
                fitted = fitted_tag

        if not fitted:
            if msg is not None:
                raise ValueError(msg)
            else:
                raise ValueError(f'{type(obj).__name__} is not fitted.')

        return fn(obj, *args, **kwargs)

    return check_and_call


def check_fitted_(attr_name='_is_fitted', check=None, msg=None):
    return partial(check_fitted, attr_name=attr_name, check=check, msg=msg)
