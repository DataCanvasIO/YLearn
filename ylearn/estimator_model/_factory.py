import pandas as pd

from ylearn import sklearn_ex as skex
from ylearn.utils import const, logging, to_repr, to_snake_case

logger = logging.get_logger(__name__)

ESTIMATOR_FACTORIES = {}


def register(name=None):
    def wrap(cls):
        assert issubclass(cls, BaseEstimatorFactory)

        if name is not None:
            tag = name
        else:
            tag = cls.__name__
            if tag.endswith('Factory'):
                tag = tag[:-7]
            tag = to_snake_case(tag)

        assert tag not in ESTIMATOR_FACTORIES.keys()

        ESTIMATOR_FACTORIES[tag] = cls
        return cls

    return wrap


class BaseEstimatorFactory:
    def __call__(self, data, outcome, treatment, y_task, x_task,
                 adjustment=None, covariate=None, instrument=None, random_state=None):
        raise NotImplemented()

    @staticmethod
    def _model(data, estimator, task, random_state=None, **kwargs):
        assert task is not None
        assert estimator is not None

        return skex.general_estimator(data, task=task, estimator=estimator, random_state=random_state, **kwargs)

    def _cf_fold(self, data):
        cf_fold = getattr(self, 'cf_fold', None)
        if cf_fold is not None:
            return cf_fold

        size = data.shape[0]
        if size < 3000:
            return 1
        elif size < 10000:
            return 3
        else:
            return 5

    def __repr__(self):
        return to_repr(self)


@register()
class DMLFactory(BaseEstimatorFactory):
    def __init__(self, y_model='rf', x_model='rf', yx_model='lr', cf_fold=None):
        self.y_model = y_model
        self.x_model = x_model
        self.yx_model = yx_model
        self.cf_fold = cf_fold

    def __call__(self, data, outcome, treatment, y_task, x_task,
                 adjustment=None, covariate=None, instrument=None, random_state=None):
        from ylearn.estimator_model.double_ml import DoubleML
        # assert adjustment is not None
        assert covariate is not None

        is_discrete_treatment = x_task if isinstance(x_task, bool) else x_task != const.TASK_REGRESSION
        is_discrete_outcome = y_task if isinstance(y_task, bool) else y_task != const.TASK_REGRESSION

        return DoubleML(
            y_model=self._model(data, task=y_task, estimator=self.y_model, random_state=random_state),
            x_model=self._model(data, task=x_task, estimator=self.x_model, random_state=random_state),
            yx_model=self._model(data, task=const.TASK_REGRESSION, estimator=self.yx_model, random_state=random_state),
            is_discrete_outcome=is_discrete_outcome,
            is_discrete_treatment=is_discrete_treatment,
            proba_output=is_discrete_outcome if is_discrete_outcome else None,
            cf_fold=self._cf_fold(data),
            random_state=random_state,
        )


@register()
class DRFactory(BaseEstimatorFactory):
    def __init__(self, y_model='gb', x_model='rf', yx_model='gb', cf_fold=None):
        self.y_model = y_model
        self.x_model = x_model
        self.yx_model = yx_model
        self.cf_fold = cf_fold

    def __call__(self, data, outcome, treatment, y_task, x_task,
                 adjustment=None, covariate=None, instrument=None, random_state=None):
        from ylearn.estimator_model import PermutedDoublyRobust

        # assert adjustment is not None
        assert x_task != const.TASK_REGRESSION, 'DoublyRobust support discrete treatment only.'

        return PermutedDoublyRobust(
            y_model=self._model(data, task=y_task, estimator=self.y_model, random_state=random_state),
            x_model=self._model(data, task=x_task, estimator=self.x_model, random_state=random_state),
            yx_model=self._model(data, task=const.TASK_REGRESSION, estimator=self.yx_model, random_state=random_state),
            cf_fold=self._cf_fold(data),
            random_state=random_state,
        )


@register()
@register(name='ml')
class MetaLearnerFactory(BaseEstimatorFactory):
    def __init__(self, learner='tlearner', model='gb', **kwargs):
        assert learner.strip().lower()[0] in {'s', 't', 'x'}

        self.learner = learner
        self.model = model
        self.options = kwargs.copy()

    def __call__(self, data, outcome, treatment, y_task, x_task,
                 adjustment=None, covariate=None, instrument=None, random_state=None):
        from ylearn.estimator_model import PermutedSLearner, PermutedTLearner, PermutedXLearner

        is_discrete_treatment = x_task if isinstance(x_task, bool) else x_task != const.TASK_REGRESSION
        is_discrete_outcome = y_task if isinstance(y_task, bool) else y_task != const.TASK_REGRESSION

        # assert adjustment is not None
        assert is_discrete_treatment, 'MetaLearner support discrete treatment only.'

        tag = self.learner.strip().lower()[0]
        learners = dict(s=PermutedSLearner, t=PermutedTLearner, x=PermutedXLearner)
        est_cls = learners[tag]
        options = dict(
            model=self._model(data, task=y_task, estimator=self.model, random_state=random_state),
            is_discrete_outcome=is_discrete_outcome,
            is_discrete_treatment=is_discrete_treatment,
            proba_output=is_discrete_outcome if is_discrete_outcome else None,
            random_state=random_state,
            # combined_treatment=False,
        )
        options.update(self.options)
        return est_cls(**options)


@register()
class SLearnerFactory(BaseEstimatorFactory):
    def __init__(self, model='gb'):
        self.model = model

    def __call__(self, data, outcome, treatment, y_task, x_task,
                 adjustment=None, covariate=None, instrument=None, random_state=None):
        from ylearn.estimator_model import PermutedSLearner

        is_discrete_treatment = x_task if isinstance(x_task, bool) else x_task != const.TASK_REGRESSION
        is_discrete_outcome = y_task if isinstance(y_task, bool) else y_task != const.TASK_REGRESSION

        assert is_discrete_treatment, 'SLearner support discrete treatment only.'

        return PermutedSLearner(
            model=self._model(data, task=y_task, estimator=self.model, random_state=random_state),
            is_discrete_outcome=is_discrete_outcome,
            is_discrete_treatment=is_discrete_treatment,
            proba_output=is_discrete_outcome if is_discrete_outcome else None,
            random_state=random_state,
            # combined_treatment=False,
        )


@register()
class TLearnerFactory(BaseEstimatorFactory):
    def __init__(self, model='gb'):
        self.model = model

    def __call__(self, data, outcome, treatment, y_task, x_task,
                 adjustment=None, covariate=None, instrument=None, random_state=None):
        from ylearn.estimator_model import PermutedTLearner

        is_discrete_treatment = x_task if isinstance(x_task, bool) else x_task != const.TASK_REGRESSION
        is_discrete_outcome = y_task if isinstance(y_task, bool) else y_task != const.TASK_REGRESSION

        assert is_discrete_treatment, 'TLearner support discrete treatment only.'

        return PermutedTLearner(
            model=self._model(data, task=y_task, estimator=self.model, random_state=random_state),
            is_discrete_outcome=is_discrete_outcome,
            is_discrete_treatment=is_discrete_treatment,
            proba_output=is_discrete_outcome if is_discrete_outcome else None,
            random_state=random_state,
            # combined_treatment=False,
        )


@register()
class XLearnerFactory(BaseEstimatorFactory):
    def __init__(self, model='gb', final_proba_model='lr'):
        self.model = model
        self.final_proba_model = final_proba_model

    def __call__(self, data, outcome, treatment, y_task, x_task,
                 adjustment=None, covariate=None, instrument=None, random_state=None):
        from ylearn.estimator_model import PermutedXLearner

        is_discrete_treatment = x_task if isinstance(x_task, bool) else x_task != const.TASK_REGRESSION
        is_discrete_outcome = y_task if isinstance(y_task, bool) else y_task != const.TASK_REGRESSION

        assert is_discrete_treatment, 'XLearner support discrete treatment only.'

        if is_discrete_outcome:
            final_proba_model = self._model(
                data, task=const.TASK_REGRESSION, estimator=self.final_proba_model, random_state=random_state)
        else:
            final_proba_model = None

        return PermutedXLearner(
            model=self._model(data, task=y_task, estimator=self.model, random_state=random_state),
            final_proba_model=final_proba_model,
            is_discrete_outcome=is_discrete_outcome,
            is_discrete_treatment=is_discrete_treatment,
            proba_output=is_discrete_outcome if is_discrete_outcome else None,
            random_state=random_state,
            # combined_treatment=False,
        )


@register()
@register(name='tree')
class CausalTreeFactory(BaseEstimatorFactory):
    def __init__(self, **kwargs):
        self.options = kwargs.copy()

    def __call__(self, data, outcome, treatment, y_task, x_task,
                 adjustment=None, covariate=None, instrument=None, random_state=None):
        from ylearn.estimator_model._permuted import PermutedCausalTree

        is_discrete_treatment = x_task if isinstance(x_task, bool) else x_task != const.TASK_REGRESSION
        is_discrete_outcome = y_task if isinstance(y_task, bool) else y_task != const.TASK_REGRESSION

        assert is_discrete_treatment, 'CausalTree support discrete treatment only.'

        options = self.options.copy()
        if random_state is not None:
            options['random_state'] = random_state
        # options['is_discrete_outcome'] = is_discrete_outcome
        # options['is_discrete_treatment'] = is_discrete_treatment

        return PermutedCausalTree(**options)


@register()
class GrfFactory(BaseEstimatorFactory):
    def __init__(self, **kwargs):
        self.options = kwargs.copy()

    def __call__(self, data, outcome, treatment, y_task, x_task,
                 adjustment=None, covariate=None, instrument=None, random_state=None):
        from ylearn.estimator_model._generalized_forest import GRForest

        is_discrete_treatment = x_task if isinstance(x_task, bool) else x_task != const.TASK_REGRESSION
        is_discrete_outcome = y_task if isinstance(y_task, bool) else y_task != const.TASK_REGRESSION

        options = self.options.copy()
        if random_state is not None:
            options['random_state'] = random_state
        options['is_discrete_outcome'] = is_discrete_outcome
        options['is_discrete_treatment'] = is_discrete_treatment

        return GRForest(**options)


@register()
@register(name='bound')
class ApproxBoundFactory(BaseEstimatorFactory):
    def __init__(self, y_model='gb', x_model='rf', x_prob=None):
        self.y_model = y_model
        self.x_model = x_model
        self.x_prob = x_prob

    def __call__(self, data, outcome, treatment, y_task, x_task,
                 adjustment=None, covariate=None, instrument=None, random_state=None):
        from ylearn.estimator_model.approximation_bound import ApproxBound

        assert covariate is not None

        return ApproxBound(
            y_model=self._model(data, task=y_task, estimator=self.y_model, random_state=random_state),
            x_model=self._model(data, task=x_task, estimator=self.x_model, random_state=random_state),
            x_prob=self.x_prob,
            is_discrete_treatment=x_task if isinstance(x_task, bool) else x_task != const.TASK_REGRESSION,
            random_state=random_state,
        )


@register()
class IVFactory(BaseEstimatorFactory):
    def __init__(self, y_model='lr', x_model='rf'):
        self.y_model = y_model
        self.x_model = x_model

    def __call__(self, data, outcome, treatment, y_task, x_task,
                 adjustment=None, covariate=None, instrument=None, random_state=None):
        from ylearn.estimator_model.iv import NP2SLS

        assert instrument is not None

        return NP2SLS(
            y_model=self._model(data, task=y_task, estimator=self.y_model, random_state=random_state),
            x_model=self._model(data, task=x_task, estimator=self.x_model, random_state=random_state),
            is_discrete_outcome=y_task if isinstance(y_task, bool) else y_task != const.TASK_REGRESSION,
            is_discrete_treatment=x_task if isinstance(x_task, bool) else x_task != const.TASK_REGRESSION,
            random_state=random_state,
        )


try:
    import torch
    from ylearn.estimator_model.deepiv import DeepIV


    class DeepIVWrapper(DeepIV):
        def fit(self, data, outcome, treatment, **kwargs):
            data = self._f64to32(data)
            return super().fit(data, outcome, treatment=treatment, **kwargs)

        def estimate(self, data=None, *args, **kwargs, ):
            if data is not None:
                data = self._f64to32(data)

            effect = super().estimate(data, *args, **kwargs)
            if isinstance(effect, torch.Tensor):
                effect = effect.detach().numpy()
            return effect

        @staticmethod
        def _f64to32(data):
            assert isinstance(data, pd.DataFrame)
            data_f64 = data.select_dtypes(include='float64')
            if len(data_f64.columns) > 0:
                data = data.copy()
                data[data.columns] = data_f64.astype('float32')
            return data

except ImportError as e:
    DeepIVWrapper = f'{e}'
    # logger.warn(f'ImportError: {e}, DeepIv is disabled.')


@register()
@register(name='div')
class DeepIVFactory(BaseEstimatorFactory):
    def __init__(self, x_net=None, y_net=None, x_hidden_d=None, y_hidden_d=None, num_gaussian=5, ):
        if isinstance(DeepIVWrapper, str):
            raise ImportError(f'{DeepIVWrapper}, install pytorch and try again.')

        self.x_net = x_net
        self.y_net = y_net
        self.x_hidden_d = x_hidden_d
        self.y_hidden_d = y_hidden_d
        self.num_gaussian = num_gaussian

    def __call__(self, data, outcome, treatment, y_task, x_task,
                 adjustment=None, covariate=None, instrument=None, random_state=None):
        assert instrument is not None

        return DeepIVWrapper(
            x_net=self.x_net,
            y_net=self.y_net,
            x_hidden_d=self.x_hidden_d,
            y_hidden_d=self.y_hidden_d,
            num_gaussian=self.num_gaussian,
            is_discrete_outcome=y_task if isinstance(y_task, bool) else y_task != const.TASK_REGRESSION,
            is_discrete_treatment=x_task if isinstance(x_task, bool) else x_task != const.TASK_REGRESSION,
            random_state=random_state,
        )


@register()
class RLossFactory(BaseEstimatorFactory):
    def __init__(self, y_model='rf', x_model='rf', cf_fold=None):
        self.y_model = y_model
        self.x_model = x_model
        self.cf_fold = cf_fold

    def __call__(self, data, outcome, treatment, y_task, x_task,
                 adjustment=None, covariate=None, instrument=None, random_state=None):
        from ylearn.estimator_model.effect_score import RLoss

        return RLoss(
            y_model=self._model(data, task=y_task, estimator=self.y_model, random_state=random_state),
            x_model=self._model(data, task=x_task, estimator=self.x_model, random_state=random_state),
            cf_fold=self._cf_fold(data),
            # is_discrete_outcome=y_task != const.TASK_REGRESSION,
            is_discrete_treatment=x_task if isinstance(x_task, bool) else x_task != const.TASK_REGRESSION,
            random_state=random_state,
        )
