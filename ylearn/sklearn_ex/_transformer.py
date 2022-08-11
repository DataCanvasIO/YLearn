import copy
import math

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler

from ylearn.utils import logging, const, infer_task_type, is_os_darwin
from ._data_cleaner import DataCleaner
from ._dataframe_mapper import DataFrameMapper

logger = logging.get_logger(__name__)


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


class FeatureImportancesSelectionTransformer(BaseEstimator):
    STRATEGY_THRESHOLD = 'threshold'
    STRATEGY_QUANTILE = 'quantile'
    STRATEGY_NUMBER = 'number'

    _default_strategy = dict(
        default_strategy=STRATEGY_THRESHOLD,
        default_threshold=0.1,
        default_quantile=0.2,
        default_number=0.8,
    )

    def __init__(self, task=None, strategy=None, threshold=None, quantile=None, number=None, data_clean=True):
        super().__init__()

        self.task = task
        self.strategy = strategy
        self.threshold = threshold
        self.quantile = quantile
        self.number = number
        self.data_clean = data_clean

        # fitted
        self.feature_names_in_ = None
        self.n_features_in_ = None
        self.feature_importances_ = None
        self.selected_features_ = None

    def fit(self, X, y):
        if self.task is None:
            self.task, _ = infer_task_type(y)

        columns_in = X.columns.to_list()
        # logger.info(f'all columns: {columns}')

        if self.data_clean:
            logger.info('data cleaning')
            kwargs = dict(replace_inf_values=np.nan, drop_label_nan_rows=True,
                          drop_constant_columns=True, drop_duplicated_columns=False,
                          drop_idness_columns=True, reduce_mem_usage=False,
                          correct_object_dtype=False, int_convert_to=None,
                          )
            dc = DataCleaner(**kwargs)
            X, y = dc.fit_transform(X, y)
            assert set(X.columns.tolist()).issubset(set(columns_in))

        preprocessor = general_preprocessor()

        if self.task != 'regression' and y.dtype != 'int':
            logger.info('label encoding')
            le = LabelEncoder()
            y = le.fit_transform(y)

        logger.info('preprocessing')
        X = preprocessor.fit_transform(X, y)

        estimator = general_estimator(X, y, task=self.task)
        logger.info(f'scoring feature_importances with {estimator} ')
        estimator.fit(X, y)
        importances = estimator.feature_importances_

        selected, unselected = \
            self._select_feature_by_importance(importances, strategy=self.strategy,
                                               threshold=self.threshold,
                                               quantile=self.quantile,
                                               number=self.number)
        columns = X.columns.to_list()
        selected = [columns[i] for i in selected]

        if len(columns) != len(columns_in):
            importances = [0.0 if c not in columns else importances[columns.index(c)] for c in columns_in]
            importances = np.array(importances)

        self.n_features_in_ = len(columns_in)
        self.feature_names_in_ = columns_in
        self.feature_importances_ = importances
        self.selected_features_ = selected

        # logger.info(f'selected columns:{self.selected_features_}')

        return self

    def transform(self, X):
        return X[self.selected_features_]

    @classmethod
    def _detect_strategy(cls, strategy, *, threshold=None, quantile=None, number=None,
                         default_strategy, default_threshold, default_quantile, default_number):
        if strategy is None:
            if threshold is not None:
                strategy = cls.STRATEGY_THRESHOLD
            elif number is not None:
                strategy = cls.STRATEGY_NUMBER
            elif quantile is not None:
                strategy = cls.STRATEGY_QUANTILE
            else:
                strategy = default_strategy

        if strategy == cls.STRATEGY_THRESHOLD:
            if threshold is None:
                threshold = default_threshold
        elif strategy == cls.STRATEGY_NUMBER:
            if number is None:
                number = default_number
        elif strategy == cls.STRATEGY_QUANTILE:
            if quantile is None:
                quantile = default_quantile
            assert 0 < quantile < 1.0
        else:
            raise ValueError(f'Unsupported strategy: {strategy}')

        return strategy, threshold, quantile, number

    @classmethod
    def _select_feature_by_importance(cls, feature_importance,
                                      strategy=None, threshold=None, quantile=None, number=None):
        assert isinstance(feature_importance, (list, tuple, np.ndarray)) and len(feature_importance) > 0

        strategy, threshold, quantile, number = cls._detect_strategy(
            strategy, threshold=threshold, quantile=quantile, number=number,
            **cls._default_strategy)

        feature_importance = np.array(feature_importance)
        idx = np.arange(len(feature_importance))

        if strategy == cls.STRATEGY_THRESHOLD:
            selected = np.where(np.where(feature_importance >= threshold, idx, -1) >= 0)[0]
        elif strategy == cls.STRATEGY_QUANTILE:
            q = np.quantile(feature_importance, quantile)
            selected = np.where(np.where(feature_importance >= q, idx, -1) >= 0)[0]
        elif strategy == cls.STRATEGY_NUMBER:
            if isinstance(number, float) and 0 < number < 1.0:
                number = math.ceil(len(feature_importance) * number)
            pos = len(feature_importance) - number
            sorted_ = np.argsort(np.argsort(feature_importance))
            selected = np.where(sorted_ >= pos)[0]
        else:
            raise ValueError(f'Unsupported strategy: {strategy}')

        unselected = list(set(range(len(feature_importance))) - set(selected))
        unselected = np.array(unselected)

        return selected, unselected


def general_preprocessor(*, number_scaler=None):
    cat_steps = [('imputer_cat', SimpleImputer(strategy='constant', fill_value='')),
                 ('encoder', SafeOrdinalEncoder())]
    num_steps = [('imputer_num', SimpleImputer(strategy='mean')),
                 # ('scaler', StandardScaler()),
                 ]
    if number_scaler is True:
        num_steps.append(('scaler', StandardScaler()))

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


def general_estimator(X, y=None, estimator=None, task=None, random_state=None, **kwargs):
    try:
        import lightgbm
        lightgbm_installed = True
    except ImportError:
        lightgbm_installed = False
    except Exception as e:
        # logger.warn(f'e')
        lightgbm_installed = False

    try:
        import xgboost
        xgboost_installed = True
    except ImportError:
        xgboost_installed = False

    def default_gbm(task_):
        est_cls = lightgbm.LGBMRegressor if task_ == const.TASK_REGRESSION else lightgbm.LGBMClassifier
        options = dict(n_estimators=50,
                       num_leaves=15,
                       max_depth=5,
                       subsample=0.5,
                       subsample_freq=1,
                       colsample_bytree=0.8,
                       reg_alpha=1,
                       reg_lambda=1,
                       importance_type='gain',
                       random_state=random_state,
                       verbose=-1,
                       **kwargs
                       )
        return est_cls(**options)

    def default_xgb(task_):
        options = dict(n_estimators=100,
                       max_depth=5,
                       min_child_weight=5,
                       learning_rate=0.1,
                       gamma=1,
                       reg_alpha=1,
                       reg_lambda=1,
                       random_state=random_state)
        if task_ == const.TASK_REGRESSION:
            est_cls = xgboost.XGBRegressor
        else:
            options['use_label_encoder'] = False
            est_cls = xgboost.XGBClassifier
        return est_cls(**options)

    def default_dt(task_):
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        est_cls = DecisionTreeRegressor if task_ == const.TASK_REGRESSION else DecisionTreeClassifier
        options = dict(min_samples_leaf=20, min_impurity_decrease=0.01, random_state=random_state, **kwargs)
        return est_cls(**options)

    def default_rf(task_):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        est_cls = RandomForestRegressor if task_ == const.TASK_REGRESSION else RandomForestClassifier
        options = dict(min_samples_leaf=20, min_impurity_decrease=0.01, random_state=random_state, **kwargs)
        return est_cls(**options)

    def default_gb(task_):
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        est_cls = GradientBoostingRegressor if task_ == const.TASK_REGRESSION else GradientBoostingClassifier
        options = dict(n_estimators=100, max_depth=100, random_state=random_state, **kwargs)
        return est_cls(**options)

    def default_lr(task_):
        from sklearn.linear_model import LinearRegression, LogisticRegression

        if task_ == const.TASK_REGRESSION:
            est_cls = LinearRegression
            options = kwargs
        else:
            est_cls = LogisticRegression
            options = dict(random_state=random_state, **kwargs)
        return est_cls(**options)

    def default_lasso(task_):
        assert task_ == const.TASK_REGRESSION
        from sklearn.linear_model import Lasso
        options = dict(random_state=random_state, **kwargs)
        return Lasso(**options)

    creators = dict(
        gbm=default_gbm,
        lgbm=default_gbm,
        lightgbm=default_gbm,
        xgb=default_xgb,
        xgboost=default_xgb,
        dt=default_dt,
        rf=default_rf,
        gb=default_gb,
        lr=default_lr,
        lasso=default_lasso,
    )

    if estimator is None:
        if xgboost_installed:
            estimator = 'xgb'
        elif lightgbm_installed and not is_os_darwin:
            estimator = 'gbm'
        else:
            estimator = 'rf'

    if task is None:
        assert y is not None, '"y" or "task" is required.'
        task = infer_task_type(y)
    elif isinstance(task, bool):  # discrete or not
        task = const.TASK_MULTICLASS if task else const.TASK_REGRESSION

    if isinstance(estimator, str) and estimator in creators.keys():
        estimator_ = creators[estimator](task)
    else:
        estimator_ = copy.deepcopy(estimator)

    return estimator_
