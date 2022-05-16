import contextlib as _contextlib
import warnings

import numpy as np
import pandas as pd


class const:
    TASK_AUTO = 'auto'
    TASK_BINARY = 'binary'
    TASK_MULTICLASS = 'multiclass'
    TASK_REGRESSION = 'regression'
    TASK_MULTILABEL = 'multilabel'


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
            dfs.append(pd.DataFrame(v, columns=[f'{k}_{i}' for i in range(v.shape[1])]))
    df = pd.concat(dfs, axis=1)
    return df


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


def view_pydot(pdot, prog='dot'):
    try:
        from IPython.display import Image, display

        img = Image(pdot.create_png(prog=prog))
        display(img)
    except Exception as e:
        warnings.warn(f'Failed to display pydot image: {e}.')


def unique(y):
    if hasattr(y, 'unique'):
        uniques = set(y.unique())
    else:
        uniques = set(y)
    return uniques


def infer_task_type(y, excludes=None):
    assert excludes is None or isinstance(excludes, (list, tuple, set))

    if len(y.shape) > 1 and y.shape[-1] > 1:
        labels = list(range(y.shape[-1]))
        task = const.TASK_MULTILABEL
        return task, labels

    uniques = unique(y)
    if uniques.__contains__(np.nan):
        uniques.remove(np.nan)
    if excludes is not None and len(excludes) > 0:
        uniques -= set(excludes)
    n_unique = len(uniques)
    labels = []

    if n_unique == 2:
        task = const.TASK_BINARY  # TASK_BINARY
        labels = sorted(uniques)
    else:
        if str(y.dtype).find('float') >= 0:
            task = const.TASK_REGRESSION
        else:
            if n_unique > 1000:
                if str(y.dtype).find('int') >= 0:
                    task = const.TASK_REGRESSION
                else:
                    raise ValueError('The number of classes exceeds 1000, please confirm whether '
                                     'your predict target is correct ')
            else:
                task = const.TASK_MULTICLASS
                labels = sorted(uniques)
    return task, labels
