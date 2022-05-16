import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from ._dataframe_mapper import DataFrameMapper


class SafeOrdinalEncoder(OrdinalEncoder):
    __doc__ = r'Adapted from sklearn OrdinalEncoder\n' + OrdinalEncoder.__doc__

    def transform(self, X, y=None):
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("Unexpected type {}".format(type(X)))

        def make_encoder(categories):
            unseen = len(categories)
            m = dict(zip(categories, range(unseen)))
            vf = np.vectorize(lambda x: m[x] if x in m.keys() else unseen)
            return vf

        values = X if isinstance(X, np.ndarray) else X.values
        encoders_ = [make_encoder(cat) for cat in self.categories_]
        result = [encoders_[i](values[:, i]) for i in range(values.shape[1])]

        if isinstance(X, pd.DataFrame):
            assert len(result) == len(X.columns)
            data = {c: result[i] for i, c in enumerate(X.columns)}
            result = pd.DataFrame(data, dtype=self.dtype)
        else:
            result = np.stack(result, axis=1)
            if self.dtype != result.dtype:
                result = result.astype(self.dtype)

        return result

    def inverse_transform(self, X):
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("Unexpected type {}".format(type(X)))

        def make_decoder(categories, dtype):
            if dtype in (np.float32, np.float64, float):
                default_value = np.nan
            elif dtype in (np.int32, np.int64, np.uint32, np.uint64, np.uint, int):
                default_value = -1
            else:
                default_value = None
                dtype = object
            unseen = len(categories)
            vf = np.vectorize(lambda x: categories[x] if unseen > x >= 0 else default_value,
                              otypes=[dtype])
            return vf

        values = X if isinstance(X, np.ndarray) else X.values
        decoders_ = [make_decoder(cat, cat.dtype) for i, cat in enumerate(self.categories_)]
        result = [decoders_[i](values[:, i]) for i in range(values.shape[1])]

        if isinstance(X, pd.DataFrame):
            assert len(result) == len(X.columns)
            data = {c: result[i] for i, c in enumerate(X.columns)}
            result = pd.DataFrame(data)
        else:
            result = np.stack(result, axis=1)

        return result


def general_preprocessor():
    cat_steps = [('imputer_cat', SimpleImputer(strategy='constant', fill_value='')),
                 ('encoder', SafeOrdinalEncoder())]
    num_steps = [('imputer_num', SimpleImputer(strategy='mean')),
                 ('scaler', StandardScaler())]

    cat_transformer = Pipeline(steps=cat_steps)
    num_transformer = Pipeline(steps=num_steps)
    bool_transformer = SafeOrdinalEncoder()

    cat_selector = make_column_selector(dtype_include=['object', 'category'])
    bool_selector = make_column_selector(dtype_include='bool')
    num_selector = make_column_selector(dtype_include='number', dtype_exclude='timedelta')

    preprocessor = DataFrameMapper(
        features=[(cat_selector, cat_transformer),
                  (bool_selector, bool_transformer),
                  (num_selector, num_transformer)],
        input_df=True,
        df_out=True)
    return preprocessor
