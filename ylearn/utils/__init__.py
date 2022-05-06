import pandas as pd


def to_df(**data):
    dfs = []
    for k, v in data.items():
        if len(v.shape) == 1:
            dfs.append(pd.Series(v, name=k))
        elif v.shape[1] == 1:
            dfs.append(pd.DataFrame(v, columns=[k]))
        else:
            dfs.append(pd.DataFrame(v, columns=[f'{k}_{i}' for i in range(v.shape[1])]))
    df = pd.concat(dfs, axis=1)
    return df
