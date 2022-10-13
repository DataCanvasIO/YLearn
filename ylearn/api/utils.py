import numpy as np
import pandas as pd

from ylearn.utils import const, infer_task_type
# noinspection PyUnresolvedReferences
from ylearn.utils import to_list, join_list


def is_empty(v):
    return v is None or len(v) == 0


def non_empty(v):
    return v is not None and len(v) > 0


def safe_remove(alist, value, copy=False):
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


def format(v, line_width=64, line_limit=3):
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


def is_number(dtype):
    return dtype.kind in {'i', 'f'}


def is_discrete(y):
    return infer_task_type(y)[0] != const.TASK_REGRESSION


def align_task_to_first(data, columns, count_limit):
    """
     select features which has similar task type with the first
    """
    selected = columns[:1]
    discrete = is_discrete(data[selected[0]])
    for x in columns[1:]:
        x_discrete = is_discrete(data[x])
        if x_discrete is discrete:
            selected.append(x)
            if len(selected) >= count_limit:
                break

    return selected


def task_tag(discrete):
    if isinstance(discrete, (np.ndarray, pd.Series)):
        discrete = is_discrete(discrete)
    return 'classification' if discrete else 'regression'


def select_by_task(data, columns, count_limit, discrete):
    """
     select features which has similar task type with the first
    """
    selected = []
    for x in columns:
        x_discrete = is_discrete(data[x])
        if x_discrete is discrete:
            selected.append(x)
            if len(selected) >= count_limit:
                break

    return selected


def cost_effect(fn_cost, test_data, effect, effect_name):
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


def cost_effect_array(fn_cost, test_data, effect_array, effect_name):
    effects = [cost_effect(fn_cost, test_data, effect_array[:, i], effect_name).reshape((-1, 1))
               for i in range(effect_array.shape[1])]
    result = np.hstack(effects)
    return result


def safe_treat_control(treatment, t_or_c, name):
    """
    Returns:
        list or None.
    """

    if t_or_c is None:
        return None

    if isinstance(t_or_c, (tuple, list)):
        assert len(t_or_c) == len(treatment), \
            f'{name} should have the same number with treatment ({treatment})'
    else:
        assert len(treatment) == 1, \
            f'{name} should be list or tuple if the number of treatment is greater than 1'
        t_or_c = [t_or_c, ]

    return t_or_c


def safe_treat_control_list(treatment, t_or_c, name):
    """
    Returns:
        nested list or None.
    """

    if t_or_c is None:
        return None

    if isinstance(t_or_c, (tuple, list)):
        if isinstance(t_or_c[0], (tuple, list)):
            assert all(len(e) == len(treatment) for e in t_or_c), \
                f'every element in {name} should have the same length with treatment'
        else:
            if len(treatment) > 1:
                if len(t_or_c) == len(treatment):
                    t_or_c = [t_or_c, ]  # to nested list
                else:
                    raise ValueError(
                        f'{name} should be nested list or tuple if the number of treatment is greater than 1'
                    )
            else:
                t_or_c = [[e] for e in t_or_c]  # to nested list
    else:
        assert len(treatment) == 1, \
            f'{name} should be nested list or tuple if the number of treatment is greater than 1'
        t_or_c = [[t_or_c, ]]  # to nested list

    return t_or_c


def select_by_treat_control(df, treatment, treat, control):
    is_treat = None
    is_control = None

    for f, t in zip(treatment, treat):
        if is_treat is None:
            is_treat = df[f] == t
        else:
            is_treat &= (df[f] == t)
    for f, t in zip(treatment, control):
        if is_control is None:
            is_control = df[f] == t
        else:
            is_control &= (df[f] == t)

    df = df[is_treat | is_control]
    return df.copy()


def transform_treat_control(encoders, *, treat, control):
    t_result = []
    c_result = []
    for e_, t_, c_ in zip(encoders, treat, control):
        t_encoded, c_encoded = e_.transform([t_, c_]).tolist()
        t_result.append(t_encoded)
        c_result.append(c_encoded)
    return tuple(t_result), tuple(c_result)


def encode_treat_control(df, *, treatment, treat, control, treat_value=1, control_value=0):
    def _encode(row):
        if all(map(lambda _: row[_[0]] == _[1], zip(treatment, treat))):
            return treat_value
        elif all(map(lambda _: row[_[0]] == _[1], zip(treatment, control))):
            return control_value
        else:
            return np.nan

    result = df.apply(_encode, axis=1)
    return result
