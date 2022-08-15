import numpy as np
import pandas as pd

from ylearn.utils import const, infer_task_type


def _to_list(v, name=None):
    if v is None or isinstance(v, (list, tuple)):
        pass
    elif isinstance(v, str):
        v = [s.strip() for s in v.split(',')]
        v = [s for s in v if len(s) > 0]
    else:
        tag = name if name is not None else 'value'
        raise ValueError(f'Unexpected {tag}: {v}')

    return v


def _join_list(*args):
    r = []
    for a in args:
        if a is None:
            pass
        elif isinstance(a, list):
            r += a
        elif isinstance(a, tuple):
            r += list(a)
        else:
            r += _to_list(a)
    return r


def _empty(v):
    return v is None or len(v) == 0


def _not_empty(v):
    return v is not None and len(v) > 0


def _safe_remove(alist, value, copy=False):
    assert alist is None or isinstance(alist, list)

    if alist is not None:
        if copy:
            alist = alist.copy()

        if isinstance(value, (list, tuple)):
            for v in value:
                if v in alist:
                    alist.remove(v)
        elif value in alist:
            alist.remove(value)

    return alist


def _format(v, line_width=64, line_limit=3):
    if isinstance(v, (list, tuple)):
        lines = []
        line = ''
        for vi in v:
            if len(line) >= line_width:
                if len(lines) + 1 >= line_limit:
                    line += '...'
                    break
                else:
                    lines.append(line)
                    line = ''  # reset new line
            line += f', {vi}' if line else f'{vi}'
        lines.append(line)
        r = ',\n'.join(lines)
    else:
        r = f'{v}'

    return r


def _is_number(dtype):
    return dtype.kind in {'i', 'f'}


def _is_discrete(y):
    return infer_task_type(y)[0] != const.TASK_REGRESSION


def _align_task_to_first(data, columns, count_limit):
    """
     select features which has similar task type with the first
    """
    selected = columns[:1]
    discrete = _is_discrete(data[selected[0]])
    for x in columns[1:]:
        x_discrete = _is_discrete(data[x])
        if x_discrete is discrete:
            selected.append(x)
            if len(selected) >= count_limit:
                break

    return selected


def _task_tag(discrete):
    if isinstance(discrete, (np.ndarray, pd.Series)):
        discrete = _is_discrete(discrete)
    return 'classification' if discrete else 'regression'


def _select_by_task(data, columns, count_limit, discrete):
    """
     select features which has similar task type with the first
    """
    selected = []
    for x in columns:
        x_discrete = _is_discrete(data[x])
        if x_discrete is discrete:
            selected.append(x)
            if len(selected) >= count_limit:
                break

    return selected


def _cost_effect(fn_cost, test_data, effect, effect_name):
    assert test_data is None or isinstance(test_data, pd.DataFrame)

    if isinstance(effect, np.ndarray):
        if effect.ndim > 1:
            effect = effect.ravel()
    elif isinstance(effect, pd.Series):
        effect = effect.values
    elif isinstance(effect, pd.DataFrame):
        assert effect.shape[1] == 1
        effect = effect.values.ravel()

    if test_data is None:
        df = pd.DataFrame({effect_name: effect})
    else:
        assert len(test_data) == len(effect)
        df = test_data.copy()
        df[effect_name] = effect

    r = df.apply(fn_cost, axis=1)
    if isinstance(r, (pd.Series, pd.DataFrame)):
        r = r.values

    return r
