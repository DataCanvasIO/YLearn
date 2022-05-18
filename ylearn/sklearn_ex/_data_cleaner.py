# -*- coding:utf-8 -*-
"""

"""
import copy

import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector

from ylearn.utils import logging

logger = logging.get_logger(__name__)

column_object = make_column_selector(dtype_include=['object'])
column_object_category_bool_int = make_column_selector(
    dtype_include=['object', 'category', 'bool',
                   'int', 'int8', 'int16', 'int32', 'int64',
                   'uint', 'uint8', 'uint16', 'uint32', 'uint64'])
column_int = make_column_selector(
    dtype_include=['int', 'int8', 'int16', 'int32', 'int64',
                   'uint', 'uint8', 'uint16', 'uint32', 'uint64'])


class _CleanerHelper:
    @staticmethod
    def reduce_mem_usage(df, excludes=None):
        """
        Adaption from :https://blog.csdn.net/xckkcxxck/article/details/88170281
        """
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024 ** 2
        for col in df.columns:
            if excludes is not None and col in excludes:
                continue
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        end_mem = df.memory_usage().sum() / 1024 ** 2
        if logger.is_info_enabled():
            logger.info('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'
                        .format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    @staticmethod
    def replace_nan_chars(X, nan_chars):
        return X.replace(nan_chars, np.nan)

    def correct_object_dtype(self, X, df_meta=None, excludes=None):
        if df_meta is None:
            Xt = X[[c for c in X.columns.to_list() if c not in excludes]] if excludes else X
            cats = []

            cons = [c for c in column_object(Xt) if c not in cats]
            df_meta = {'object': cats, 'float': cons}

        X = self._correct_object_dtype_as(X, df_meta)

        return X

    def _correct_object_dtype_as(self, X, df_meta):
        for dtype, columns in df_meta.items():
            columns = [c for c in columns if str(X[c].dtype) != dtype]
            if len(columns) == 0:
                continue

            if dtype in ['object', 'str']:
                X[columns] = X[columns].astype(dtype)
            else:
                for col in columns:
                    try:
                        if str(X[col].dtype) != str(dtype):
                            X[col] = X[col].astype(dtype)
                    except Exception as e:
                        if logger.is_debug_enabled():
                            logger.debug(f'Correct object column [{col}] as {dtype} failed. {e}')

        return X

    def drop_duplicated_columns(self, X, excludes=None):
        duplicates = self._get_duplicated_columns(X)
        dup_cols = [i for i, v in duplicates.items() if v and (excludes is None or i not in excludes)]
        if len(dup_cols) > 0:
            columns = [c for c in X.columns.to_list() if c not in dup_cols]
            X = X[columns]

        return X, dup_cols

    def drop_constant_columns(self, X, excludes=None):
        nunique = self._get_df_uniques(X)
        const_cols = [i for i, v in nunique.items() if v <= 1 and (excludes is None or i not in excludes)]
        if len(const_cols) > 0:
            columns = [c for c in X.columns.to_list() if c not in const_cols]
            X = X[columns]
        return X, const_cols

    def drop_idness_columns(self, X, excludes=None):
        cols = column_object_category_bool_int(X)
        if len(cols) <= 0:
            return X, []

        X_ = X[cols]
        nunique = self._get_df_uniques(X_)
        rows = len(X)
        threshold = 0.99  # cfg.idness_threshold
        dropped = [c for c, n in nunique.items() if n / rows > threshold and (excludes is None or c not in excludes)]
        if len(dropped) > 0:
            columns = [c for c in X.columns.to_list() if c not in dropped]
            X = X[columns]
        return X, dropped

    @staticmethod
    def _get_df_uniques(df):
        return df.nunique(dropna=True)

    @staticmethod
    def _get_duplicated_columns(df):
        return df.T.duplicated()


class DataCleaner:
    def __init__(self, nan_chars=None, correct_object_dtype=True, drop_constant_columns=True,
                 drop_duplicated_columns=False, drop_label_nan_rows=True, drop_idness_columns=True,
                 replace_inf_values=np.nan, drop_columns=None, reserve_columns=None,
                 reduce_mem_usage=False, int_convert_to='float'):
        self.nan_chars = nan_chars
        self.correct_object_dtype = correct_object_dtype
        self.drop_constant_columns = drop_constant_columns
        self.drop_label_nan_rows = drop_label_nan_rows
        self.drop_idness_columns = drop_idness_columns
        self.replace_inf_values = replace_inf_values
        self.drop_columns = drop_columns
        self.reserve_columns = reserve_columns
        self.drop_duplicated_columns = drop_duplicated_columns
        self.reduce_mem_usage = reduce_mem_usage
        self.int_convert_to = int_convert_to

        # fitted
        self.df_meta_ = None
        self.columns_ = None
        self.dropped_constant_columns_ = None
        self.dropped_idness_columns_ = None
        self.dropped_duplicated_columns_ = None

    def get_params(self):
        return {
            'nan_chars': self.nan_chars,
            'correct_object_dtype': self.correct_object_dtype,
            'drop_constant_columns': self.drop_constant_columns,
            'drop_label_nan_rows': self.drop_label_nan_rows,
            'drop_idness_columns': self.drop_idness_columns,
            # 'replace_inf_values': self.replace_inf_values,
            'drop_columns': self.drop_columns,
            'reserve_columns': self.reserve_columns,
            'drop_duplicated_columns': self.drop_duplicated_columns,
            'reduce_mem_usage': self.reduce_mem_usage,
            'int_convert_to': self.int_convert_to
        }

    @staticmethod
    def get_helper(X, y):
        return _CleanerHelper()

    @staticmethod
    def _drop_columns(X, cols):
        if cols is None or len(cols) <= 0:
            return X
        X = X[[c for c in X.columns.to_list() if c not in cols]]
        return X

    def clean_data(self, X, y, *, df_meta=None, reduce_mem_usage):
        y_name = '__tabular-toolbox__Y__'

        if y is not None:
            X[y_name] = y

        helper = self.get_helper(X, y)

        if self.nan_chars is not None:
            logger.debug(f'replace chars{self.nan_chars} to NaN')
            # X = X.replace(self.nan_chars, np.nan)
            X = helper.replace_nan_chars(X, self.nan_chars)

        if y is not None:
            if self.drop_label_nan_rows:
                logger.debug('clean the rows which label is NaN')
                X = X.dropna(subset=[y_name])
            y = X.pop(y_name)

        if self.drop_columns is not None:
            logger.debug(f'drop columns:{self.drop_columns}')
            X = self._drop_columns(X, self.drop_columns)

        if self.drop_duplicated_columns:
            if self.dropped_duplicated_columns_ is not None:
                X = self._drop_columns(X, self.dropped_duplicated_columns_)
            else:
                X, self.dropped_duplicated_columns_ = helper.drop_duplicated_columns(X, self.reserve_columns)
            logger.info(f'drop duplicated columns: "{self.dropped_duplicated_columns_}')

        if self.drop_idness_columns:
            if self.dropped_idness_columns_ is not None:
                X = self._drop_columns(X, self.dropped_idness_columns_)
            else:
                X, self.dropped_idness_columns_ = helper.drop_idness_columns(X, self.reserve_columns)
            logger.debug(f'drop idness columns: {self.dropped_idness_columns_}')

        if self.drop_constant_columns:
            if self.dropped_constant_columns_ is not None:
                X = self._drop_columns(X, self.dropped_constant_columns_)
            else:
                X, self.dropped_constant_columns_ = helper.drop_constant_columns(X, self.reserve_columns)
            logger.debug(f'drop constant columns: {self.dropped_constant_columns_}')

        if self.replace_inf_values is not None:
            logger.info(f'replace [inf,-inf] to {self.replace_inf_values}')
            int_cols = column_int(X)
            if len(int_cols) > 0:
                columns = [c for c in X.columns.to_list() if c not in int_cols]
                if len(columns) > 0:
                    X[columns] = X[columns].replace([np.inf, -np.inf], self.replace_inf_values)
            else:
                X = X.replace([np.inf, -np.inf], self.replace_inf_values)

        if self.correct_object_dtype:
            logger.debug('correct data type for object columns.')
            # for col in column_object(X):
            #     try:
            #         X[col] = X[col].astype('float')
            #     except Exception as e:
            #         logger.error(f'Correct object column [{col}] failed. {e}')
            X = helper.correct_object_dtype(X, df_meta, excludes=self.reserve_columns)

        if self.int_convert_to is not None:
            int_cols = column_int(X)
            if self.reserve_columns:
                int_cols = list(filter(lambda _: _ not in self.reserve_columns, int_cols))
            if len(int_cols) > 0:
                logger.info(f'convert int type to {self.int_convert_to}, {int_cols}')
                X[int_cols] = X[int_cols].astype(self.int_convert_to)

        o_cols = column_object(X)
        if self.reserve_columns:
            o_cols = list(filter(lambda _: _ not in self.reserve_columns, o_cols))
        if o_cols:
            logger.info(f'convert object type to str, {o_cols}')
            X[o_cols] = X[o_cols].astype('str')

        if reduce_mem_usage:
            logger.info('try reduce memory usage')
            helper.reduce_mem_usage(X, excludes=self.reserve_columns)

        return X, y

    @staticmethod
    def _copy(X, y):
        if isinstance(X, pd.DataFrame):
            X = copy.deepcopy(X)
        else:
            X = X.copy()

        if y is not None:
            if isinstance(y, (np.ndarray, pd.Series, pd.DataFrame)):
                y = copy.deepcopy(y)
            else:
                y = y.copy()

        return X, y

    def fit_transform(self, X, y=None, copy_data=True):
        if copy_data:
            X, y = self._copy(X, y)

        X, y = self.clean_data(X, y, reduce_mem_usage=self.reduce_mem_usage)

        logger.debug('collect meta info from data')
        df_meta = {}
        for col_info in zip(X.columns.to_list(), X.dtypes):
            dtype = str(col_info[1])
            if df_meta.get(dtype) is None:
                df_meta[dtype] = []
            df_meta[dtype].append(col_info[0])

        logger.info(f'cleaned dataframe meta:{df_meta}')
        self.df_meta_ = df_meta
        self.columns_ = X.columns.to_list()

        return X, y

    def transform(self, X, y=None, copy_data=True):
        if copy_data:
            X, y = self._copy(X, y)

        orig_columns = X.columns.to_list()
        X, y = self.clean_data(X, y, df_meta=self.df_meta_, reduce_mem_usage=False)
        # if self.df_meta_ is not None:
        #     logger.debug('processing with meta info')
        #     all_cols = []
        #     for dtype, cols in self.df_meta_.items():
        #         all_cols += cols
        #         X[cols] = X[cols].astype(dtype)
        #     drop_cols = set(X.columns.to_list()) - set(all_cols)
        #     X = X[all_cols]
        #     logger.debug(f'droped columns:{drop_cols}')

        X = X[self.columns_]
        if logger.is_info_enabled():
            dropped = [c for c in orig_columns if c not in self.columns_]
            logger.info(f'drop columns: {dropped}')

        if y is None:
            return X
        else:
            return X, y

    def append_drop_columns(self, columns):
        if self.df_meta_ is None:
            if self.drop_columns is None:
                self.drop_columns = []
            self.drop_columns = list(set(self.drop_columns + columns))
        else:
            meta = {}
            for dtype, cols in self.df_meta_.items():
                meta[dtype] = [c for c in cols if c not in columns]
            self.df_meta_ = meta

        self.columns_ = [c for c in self.columns_ if c not in columns]

    def _repr_html_(self):
        cleaner_info = [
            ('Meta', self.df_meta_),
            ('Dropped constant columns', self.dropped_constant_columns_),
            ('Dropped idness columns', self.dropped_idness_columns_),
            ('Dropped duplicated columns', self.dropped_duplicated_columns_),
            ('-------------params-------------', '-------------values-------------'),
            ('nan_chars', self.nan_chars),
            ('correct_object_dtype', self.correct_object_dtype),
            ('drop_constant_columns', self.drop_constant_columns),
            ('drop_label_nan_rows', self.drop_label_nan_rows),
            ('drop_idness_columns', self.drop_idness_columns),
            ('replace_inf_values', self.replace_inf_values),
            ('drop_columns', self.drop_columns),
            ('reserve_columns', self.reserve_columns),
            ('drop_duplicated_columns', self.drop_duplicated_columns),
            ('reduce_mem_usage', self.reduce_mem_usage),
            ('int_convert_to', self.int_convert_to),
        ]

        html = pd.DataFrame(cleaner_info, columns=['key', 'value'])._repr_html_()
        return html
