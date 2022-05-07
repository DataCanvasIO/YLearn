import pandas as _pd


def to_df(**data):
    dfs = []
    for k, v in data.items():
        if v is None:
            pass  # ignore the item
        elif len(v.shape) == 1:
            dfs.append(_pd.Series(v, name=k))
        elif v.shape[1] == 1:
            dfs.append(_pd.DataFrame(v, columns=[k]))
        else:
            dfs.append(_pd.DataFrame(v, columns=[f'{k}_{i}' for i in range(v.shape[1])]))
    df = _pd.concat(dfs, axis=1)
    return df
