# -*- coding:utf-8 -*-
"""
Clone from Hypernets: https://github.com/DataCanvasIO/Hypernets/blob/master/hypernets/tabular/dataframe_mapper.py
"""
import hashlib

import numpy as np
import pandas as pd
from scipy import sparse as _sparse
from sklearn.base import BaseEstimator
from sklearn.pipeline import _name_estimators, Pipeline
from sklearn.utils import tosequence

from ylearn.utils import logging, context

logger = logging.get_logger(__name__)


def _call_fit(fit_method, X, y=None, **kwargs):
    """
    helper function, calls the fit or fit_transform method with the correct
    number of parameters

    fit_method: fit or fit_transform method of the transformer
    X: the data to fit
    y: the target vector relative to X, optional
    kwargs: any keyword arguments to the fit method

    return: the result of the fit or fit_transform method

    WARNING: if this function raises a TypeError exception, test the fit
    or fit_transform method passed to it in isolation as _call_fit will not
    distinguish TypeError due to incorrect number of arguments from
    other TypeError
    """
    try:
        return fit_method(X, y, **kwargs)
    except TypeError:
        # fit takes only one argument
        return fit_method(X, **kwargs)


def _hash(data):
    m = hashlib.md5()
    m.update(data)
    return m.hexdigest()


class TransformerPipeline(Pipeline):
    """
    Pipeline that expects all steps to be transformers taking a single X
    argument, an optional y argument, and having fit and transform methods.

    Code is copied from sklearn's Pipeline
    """

    def __init__(self, steps):
        names, estimators = zip(*steps)
        if len(dict(steps)) != len(steps):
            raise ValueError(
                "Provided step names are not unique: %s" % (names,))

        # shallow copy of steps
        self.steps = tosequence(steps)
        estimator = estimators[-1]

        for e in estimators:
            if (not (hasattr(e, "fit") or hasattr(e, "fit_transform")) or not
            hasattr(e, "transform")):
                raise TypeError("All steps of the chain should "
                                "be transforms and implement fit and transform"
                                " '%s' (type %s) doesn't)" % (e, type(e)))

        if not hasattr(estimator, "fit"):
            raise TypeError("Last step of chain should implement fit "
                            "'%s' (type %s) doesn't)"
                            % (estimator, type(estimator)))

    def _pre_transform(self, X, y=None, **fit_params):
        fit_params_steps = dict((step, {}) for step, _ in self.steps)
        for pname, pval in fit_params.items():
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        Xt = X
        for name, transform in self.steps[:-1]:
            if hasattr(transform, "fit_transform"):
                Xt = _call_fit(transform.fit_transform,
                               Xt, y, **fit_params_steps[name])
            else:
                Xt = _call_fit(transform.fit,
                               Xt, y, **fit_params_steps[name]).transform(Xt)
        return Xt, fit_params_steps[self.steps[-1][0]]

    def fit(self, X, y=None, **fit_params):
        Xt, fit_params = self._pre_transform(X, y, **fit_params)
        _call_fit(self.steps[-1][-1].fit, Xt, y, **fit_params)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        Xt, fit_params = self._pre_transform(X, y, **fit_params)
        if hasattr(self.steps[-1][-1], 'fit_transform'):
            return _call_fit(self.steps[-1][-1].fit_transform,
                             Xt, y, **fit_params)
        else:
            return _call_fit(self.steps[-1][-1].fit,
                             Xt, y, **fit_params).transform(Xt)


def make_transformer_pipeline(*steps):
    """Construct a TransformerPipeline from the given estimators.
    """
    return TransformerPipeline(_name_estimators(steps))


def _get_feature_names(estimator, columns=None):
    """
    Attempt to extract feature names based on a given estimator
    """
    if hasattr(estimator, 'get_feature_names'):
        return estimator.get_feature_names(columns)
    if hasattr(estimator, 'classes_'):
        return estimator.classes_
    return None


class DataFrameMapper(BaseEstimator):
    """
    Map Pandas data frame column subsets to their own sklearn transformation.

    Parameters:
    ----------
    features :  a list of tuples with features definitions.
                The first element is the pandas column selector. This can
                be a string (for one column) or a list of strings.
                The second element is an object that supports
                sklearn's transform interface, or a list of such objects.
                The third element is optional and, if present, must be
                a dictionary with the options to apply to the
                transformation. Example: {'alias': 'day_of_week'}

    default :   default transformer to apply to the columns not
                explicitly selected in the mapper. If False (default),
                discard them. If None, pass them through untouched. Any
                other transformer will be applied to all the unselected
                columns as a whole, taken as a 2d-array.

    df_out :    return a pandas data frame, with each column named using
                the pandas column that created it (if there's only one
                input and output) or the input columns joined with '_'
                if there's multiple inputs, and the name concatenated with
                '_1', '_2' etc if there's multiple outputs.

    input_df :  If ``True`` pass the selected columns to the transformers
                as a pandas DataFrame or Series. Otherwise pass them as a
                numpy array. Defaults to ``False``.

    Attributes
    ----------
    fitted_features_ : list of tuple(column_name list, fitted transformer, options).
    """

    def __init__(self, features, default=False, df_out=False, input_df=False, df_out_dtype_transforms=None):
        self.features = features
        self.default = default
        self.df_out = df_out
        self.input_df = input_df
        self.df_out_dtype_transforms = df_out_dtype_transforms

        # fitted
        self.feature_names_in_ = None
        self.n_features_in_ = None
        self.fitted_features_ = None

    @staticmethod
    def _build_transformer(transformers):
        if isinstance(transformers, list):
            transformers = make_transformer_pipeline(*transformers)
        return transformers

    def _build_feature(self, columns, transformers, options={}):
        return columns, self._build_transformer(transformers), options

    def _build(self, features, default):
        if isinstance(features, list):
            built_features = [self._build_feature(*f) for f in features]
        else:
            built_features = features
        built_default = self._build_transformer(default)

        return built_features, built_default

    def fit(self, X, y=None):
        built_features, built_default = self._build(self.features, self.default)

        columns_in = X.columns.to_list()
        fitted_features = []
        selected_columns = []

        for columns_def, transformers, options in built_features:
            logger.debug(f'columns:({columns_def}), transformers:({transformers}), options:({options})')
            if callable(columns_def):
                columns = columns_def(X)
            elif isinstance(columns_def, str):
                columns = [columns_def]
            else:
                columns = columns_def

            if isinstance(columns, (list, tuple)):
                columns = [c for c in columns if c not in selected_columns]

            fitted_features.append((columns, transformers, options))
            if columns is None or len(columns) <= 0:
                continue

            selected_columns += columns
            if transformers is not None:
                input_df = options.get('input_df', self.input_df)
                with context(columns):
                    Xt = self._get_col_subset(X, columns, input_df)
                    if logger.is_debug_enabled():
                        msg = f'fit {type(transformers).__name__}, X shape:{X.shape}'
                        if hasattr(transformers, 'steps'):
                            msg += f", steps:{[s[0] for s in transformers.steps]}"
                        logger.debug(msg)
                    _call_fit(transformers.fit, Xt, y)
                    # print(f'{transformers}:{Xt.dtypes}')

        # handle features not explicitly selected
        if built_default is not False and len(X.columns) > len(selected_columns):
            unselected_columns = [c for c in X.columns.to_list() if c not in selected_columns]
            if built_default is not None:
                with context(unselected_columns):
                    Xt = self._get_col_subset(X, unselected_columns, self.input_df)
                    _call_fit(built_default.fit, Xt, y)
            fitted_features.append((unselected_columns, built_default, {}))

        self.feature_names_in_ = columns_in
        self.n_features_in_ = len(columns_in)
        self.fitted_features_ = fitted_features

        return self

    def transform(self, X):
        selected_columns = []
        transformed_columns = []
        extracted = []

        for columns, transformers, options in self.fitted_features_:
            if columns is None or len(columns) < 1:
                continue
            selected_columns += columns

            input_df = options.get('input_df', self.input_df)
            alias = options.get('alias')

            Xt = self._get_col_subset(X, columns, input_df)
            if transformers is not None:
                with context(columns):
                    if logger.is_debug_enabled():
                        msg = f'transform {type(transformers).__name__}, X shape:{X.shape}'
                        if hasattr(transformers, 'steps'):
                            msg += f", steps:{[s[0] for s in transformers.steps]}"
                        logger.debug(msg)
                    # print(f'before ---- {transformers}:{Xt.dtypes}')
                    Xt = transformers.transform(Xt)
                    # print(f'after ---- {transformers}:{pd.DataFrame(Xt).dtypes}')

            extracted.append(self._fix_feature(Xt))
            transformed_columns += self._get_names(columns, transformers, Xt, alias)

        return self._to_transform_result(X, extracted, transformed_columns)

    def fit_transform(self, X, y=None, *fit_args):
        columns_in = X.columns.to_list()
        fitted_features = []
        selected_columns = []
        transformed_columns = []
        extracted = []

        built_features, built_default = self._build(self.features, self.default)
        for columns_def, transformers, options in built_features:
            if callable(columns_def):
                columns = columns_def(X)
            elif isinstance(columns_def, str):
                columns = [columns_def]
            else:
                columns = columns_def
            if isinstance(columns, (list, tuple)) and len(set(selected_columns).intersection(set(columns))) > 0:
                columns = [c for c in columns if c not in selected_columns]

            if columns is None or len(columns) < 1:
                continue

            fitted_features.append((columns, transformers, options))
            selected_columns += columns
            if logger.is_debug_enabled():
                logger.debug(f'fit_transform {len(columns)} columns with:\n{transformers}')

            input_df = options.get('input_df', self.input_df)
            alias = options.get('alias')

            Xt = self._get_col_subset(X, columns, input_df)
            if transformers is not None:
                with context(columns):
                    if logger.is_debug_enabled():
                        msg = f'fit_transform {type(transformers).__name__}, X shape:{X.shape}'
                        if hasattr(transformers, 'steps'):
                            msg += f", steps:{[s[0] for s in transformers.steps]}"
                        logger.debug(msg)
                    if hasattr(transformers, 'fit_transform'):
                        Xt = _call_fit(transformers.fit_transform, Xt, y)
                    else:
                        _call_fit(transformers.fit, Xt, y)
                        Xt = transformers.transform(Xt)

            extracted.append(self._fix_feature(Xt))
            if logger.is_debug_enabled():
                logger.debug(f'columns:{len(columns)}')
            transformed_columns += self._get_names(columns, transformers, Xt, alias)
            if logger.is_debug_enabled():
                logger.debug(f'transformed_names_:{len(transformed_columns)}')

        # handle features not explicitly selected
        if built_default is not False and len(X.columns) > len(selected_columns):
            unselected_columns = [c for c in X.columns.to_list() if c not in selected_columns]
            Xt = self._get_col_subset(X, unselected_columns, self.input_df)
            if built_default is not None:
                with context(unselected_columns):
                    if hasattr(built_default, 'fit_transform'):
                        Xt = _call_fit(built_default.fit_transform, Xt, y)
                    else:
                        _call_fit(built_default.fit, Xt, y)
                        Xt = built_default.transform(Xt)
                transformed_columns += self._get_names(unselected_columns, built_default, Xt)
            else:
                # if not applying a default transformer, keep column names unmodified
                transformed_columns += unselected_columns
            extracted.append(self._fix_feature(Xt))

            fitted_features.append((unselected_columns, built_default, {}))

        self.feature_names_in_ = columns_in
        self.n_features_in_ = len(columns_in)
        self.fitted_features_ = fitted_features

        return self._to_transform_result(X, extracted, transformed_columns)

    @staticmethod
    def _get_col_subset(X, cols, input_df=False):
        t = X[cols]
        if input_df:
            return t
        else:
            return t.values

    def _get_names(self, columns, transformer, x, alias=None):
        """
        Return verbose names for the transformed columns.

        columns       name (or list of names) of the original column(s)
        transformer   transformer - can be a TransformerPipeline
        x             transformed columns (numpy.ndarray)
        alias         base name to use for the selected columns
        """
        # logger.debug(
        #     f'get_names: {isinstance(columns, list)}, len(columns):{len(columns)} columns:{columns}, alias:{alias}')
        if alias is not None:
            name = alias
        elif isinstance(columns, list):
            name = '_'.join(map(str, columns))
            if len(name) > 64:
                name = name[:32] + _hash(name.encode('utf-8'))
        else:
            name = columns
        num_cols = x.shape[1] if len(x.shape) > 1 else 1
        if num_cols > 1:
            # If there are as many columns as classes in the transformer,
            # infer column names from classes names.

            # If we are dealing with multiple transformers for these columns
            # attempt to extract the names from each of them, starting from the
            # last one
            # logger.debug(f'transformer:{transformer}')
            if isinstance(transformer, (TransformerPipeline, Pipeline)):
                inverse_steps = transformer.steps[::-1]
                # estimators = (estimator for _, estimator in inverse_steps)
                # names_steps = (_get_feature_names(e, columns) for e in estimators)
                # names = next((n for n in names_steps if n is not None), None)
                names = None
                for _, estimator in inverse_steps:
                    names = _get_feature_names(estimator, columns)
                    if names is not None and len(names) == num_cols:
                        break
            else:  # Otherwise use the only estimator present
                names = _get_feature_names(transformer, columns)

            if names is None and len(columns) == num_cols:
                names = list(columns)

            if names is not None and len(names) == num_cols:
                names = list(names)  # ['%s_%s' % (name, o) for o in names]
            else:  # otherwise, return name concatenated with '_1', '_2', etc.
                names = [name + '_' + str(o) for o in range(num_cols)]

            if logger.is_debug_enabled():
                # logger.debug(f'names:{names}')
                logger.debug(f'transformed names:{len(names)}')
            return names
        else:
            return [name]

    @staticmethod
    def _fix_feature(fea):
        if _sparse.issparse(fea):
            fea = fea.toarray()

        if len(fea.shape) == 1:
            """
            Convert 1-dimensional arrays to 2-dimensional column vectors.
            """
            fea = np.array([fea]).T

        return fea

    def _to_transform_result(self, X, extracted, transformed_columns):
        if extracted is None or len(extracted) == 0:
            raise ValueError("No data output, ??? ")

        if self.df_out:
            df = self._to_df(X, extracted, transformed_columns)
            df = self._dtype_transform(df)
            return df
        else:
            return self._hstack_array(extracted)

    @staticmethod
    def _hstack_array(extracted):
        stacked = np.hstack(extracted)
        return stacked

    def _to_df(self, X, extracted, columns):
        dfs = [pd.DataFrame(arr, index=None).reset_index(drop=True) for arr in extracted]
        df = pd.concat(dfs, axis=1, ignore_index=True) if len(dfs) > 1 else dfs[0]
        df.columns = columns
        if len(X) == len(df):
            df.index = X.index  # reuse the original index

        return df

    def _dtype_transform(self, df_out):
        if self.df_out_dtype_transforms is not None:
            for columns, dtype in self.df_out_dtype_transforms:
                if callable(columns):
                    columns = columns(df_out)
                if isinstance(columns, list) and len(columns) <= 0:
                    continue
                df_out[columns] = df_out[columns].astype(dtype)
        return df_out
