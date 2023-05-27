import copy

import torch
from torch import nn

import pyro
from pyro import distributions as dist
from pyro.distributions import constraints
from . import _base


def _tensor_X(fn):
    assert callable(fn)
    import inspect

    sig = inspect.signature(fn)
    assert 'self' in sig.parameters.keys()

    def _call(obj, X, **kwargs):
        if X is None or isinstance(X, torch.Tensor):
            pass
        elif isinstance(X, (list, tuple)):
            X = torch.hstack([x.reshape(-1, 1) for x in X])
        else:
            raise ValueError(f'')

        if X is not None:
            X = X.double()  # fixme

        return fn(obj, X, **kwargs)

    return _call


def N(*args):
    """
    to pyro variable name
    """
    return '__'.join(args)


class UncertainObject(_base.BObject):
    @staticmethod
    def param(name, init_tensor=None, constraint=constraints.real, event_dim=None):
        return pyro.param(name,
                          init_tensor=init_tensor,
                          constraint=constraint,
                          event_dim=event_dim)

    @staticmethod
    def sample(name, fn, *args, **kwargs):
        return pyro.sample(name, fn, *args, **kwargs)


###########################################################
# Linear & MLP

class Linear(UncertainObject):
    def __init__(self, name, n_inputs, n_output, random='normal', bias=True):
        assert n_inputs > 0
        assert n_output > 0
        assert random in {'normal', 'dirichlet'}

        super().__init__()

        weight_shape = [n_inputs, n_output]
        if random == 'dirichlet':
            weight = self.sample(
                N(name, 'weight'),
                dist.Dirichlet(torch.ones(weight_shape)).to_event(1))
        else:
            weight = self.sample(
                N(name, 'weight'),
                dist.Normal(torch.zeros(weight_shape), torch.ones(weight_shape)).to_event(2))

        if bias:
            if random == 'dirichlet':
                bias = self.sample(
                    N(name, 'bias'),
                    dist.Dirichlet(torch.ones([n_output, ])))
            else:
                if n_output > 1:
                    d = dist.Normal(torch.zeros([n_output, ]), torch.ones([n_output, ])).to_event(1)
                else:
                    d = dist.Normal(0.0, 1.0)
                bias = self.sample(N(name, 'bias'), d)
        else:
            bias = None

        self.name = name
        self.random = random
        self.n_inputs = n_inputs
        self.n_outputs = n_output
        self.weight = weight
        self.bias = bias

    def __call__(self, X):
        assert X.shape[1] == self.n_inputs

        X = X.double()  # fixme
        weight = self.weight.double()
        result = X @ weight
        if self.bias is not None:
            result = result + self.bias

        return result

    @classmethod
    def guide(cls, name, n_inputs, n_output, random='normal', bias=True):
        assert n_inputs > 0
        assert n_output > 0
        assert random in {'normal', 'dirichlet'}

        weight_shape = [n_inputs, n_output]
        if random == 'dirichlet':
            concentration = cls.param(
                N(name, 'weight', 'concentration'),
                torch.ones(weight_shape),
                constraint=dist.Dirichlet.arg_constraints['concentration'])
            weight = cls.sample(
                N(name, 'weight'),
                dist.Dirichlet(concentration).to_event(1))
        else:
            mean = cls.param(
                N(name, 'weight', 'mean'),
                torch.zeros(weight_shape))
            scale = cls.param(
                N(name, 'weight', 'scale'),
                torch.ones(weight_shape),
                dist.Normal.arg_constraints['scale'])
            weight = cls.sample(
                N(name, 'weight'),
                dist.Normal(mean, scale).to_event(2))

        if bias:
            if random == 'dirichlet':
                concentration = cls.param(
                    N(name, 'bias', 'concentration'),
                    torch.ones([n_output, ]),
                    constraint=dist.Dirichlet.arg_constraints['concentration'], )
                bias = cls.sample(
                    N(name, 'bias'),
                    dist.Dirichlet(concentration))
            else:
                if n_output > 1:
                    mean = cls.param(
                        N(name, 'bias', 'mean'),
                        torch.zeros([n_output, ]))
                    scale = cls.param(
                        N(name, 'bias', 'scale'),
                        torch.ones([n_output, ]),
                        dist.Normal.arg_constraints['scale'])
                    bias = cls.sample(
                        N(name, 'bias'),
                        dist.Normal(mean, scale).to_event(1))
                else:
                    mean = cls.param(
                        N(name, 'bias', 'mean'),
                        torch.tensor(0.0))
                    scale = cls.param(
                        N(name, 'bias', 'scale'),
                        torch.tensor(1),
                        dist.Normal.arg_constraints['scale'])
                    bias = cls.sample(
                        N(name, 'bias'),
                        dist.Normal(mean, scale))
        else:
            bias = None

        result = {N(name, 'weight'): weight, }
        if bias is not None:
            result[N(name, 'bias')] = bias

        return result


class MLP(UncertainObject):
    def __init__(self, name, n_inputs, n_output, hidden_dims=None, random='normal', bias=True):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [n_inputs]

        dims = [n_inputs, *hidden_dims, n_output]
        layers = []
        n_layers = len(dims)
        for i, (n_inputs, n_outputs) in enumerate(zip(dims, dims[1:])):
            layers.append(Linear(N(name, f'L{i}'), n_inputs, n_outputs, random=random, bias=bias))
            if i < n_layers - 2:
                layers.append(nn.ELU())

        self.layers = layers
        self.name = name
        self.dims = dims

    def __call__(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

    @classmethod
    def guide(cls, name, n_inputs, n_output, hidden_dims=None, random='normal', bias=True):
        if hidden_dims is None:
            hidden_dims = [n_inputs]
        dims = [n_inputs, *hidden_dims, n_output]

        result = {}
        for i, (n_inputs, n_outputs) in enumerate(zip(dims, dims[1:])):
            result.update(Linear.guide(N(name, f'L{i}'), n_inputs, n_outputs, random=random, bias=bias))

        return result


###########################################################
# Modules


class BModule(UncertainObject):
    """
    Module to generate node from inputs.
    """

    def __init__(self, name, state, n_inputs):
        assert n_inputs >= 0
        super().__init__()

        self.name = name
        self.state = copy.copy(state)
        self.n_inputs = n_inputs

    def __call__(self, X, obs=None):
        raise NotImplementedError()

    @property
    def n_outputs(self):
        if isinstance(self.state, _base.CategoryNodeState):
            return self.state.n_classes
        else:
            return 1

    @classmethod
    def guide(cls, name, state, n_inputs):
        raise NotImplementedError()


class NiModule(BModule):
    """
    BModule with none-inputs
    """

    def __init__(self, name, state, n_inputs=0):
        assert n_inputs == 0
        super().__init__(name, state, n_inputs)


class MiModule(BModule):
    """
    BModule with multi-inputs
    """

    def __init__(self, name, state, n_inputs):
        assert n_inputs > 0
        super().__init__(name, state, n_inputs)

    def process(self, X):
        raise NotImplementedError()


class _MiModuleMeta(type):
    """
    A metaclass to dynamic subclass a MiModule and override 'process' with Fn object.
    """

    __cls_cache = {}

    @staticmethod
    def _get_cls(mod_cls, fn_cls, fn_options):
        assert issubclass(mod_cls, MiModule)
        assert isinstance(fn_cls, type)
        assert isinstance(fn_options, dict)

        cls_cache = _MiModuleMeta.__cls_cache
        cls_name = f'{fn_cls.__name__}{mod_cls.__name__}'
        if cls_name in cls_cache.keys():
            return cls_cache[cls_name]

        class Stub(mod_cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._mi_fn = fn_cls(self.name, self.n_inputs, self.n_outputs, **fn_options)

            @_tensor_X
            def process(self, X):
                return self._mi_fn(X)

            @classmethod
            def guide(cls, name, state, n_inputs):
                if isinstance(state, _base.CategoryNodeState):
                    n_outputs = state.n_classes
                else:
                    n_outputs = 1

                result = {}
                result.update(fn_cls.guide(name, n_inputs, n_outputs, **fn_options))
                result.update(mod_cls.guide(name, state, n_inputs))

                return result

            def __repr__(self):
                r = f'{cls_name}(name=\'{self.name}\', state={self.state}, n_inputs={self.n_inputs})'
                return r

            def __reduce__(self):
                state = getattr(self, "__getstate__", self.__dict__.copy)()
                return _MiModuleMeta._new_obj, (mod_cls, fn_cls, fn_options), state

        Stub.__name__ = cls_name
        cls_cache[cls_name] = Stub
        return Stub

    @staticmethod
    def _new_obj(mod_cls, fn_cls, fn_options):
        cls = _MiModuleMeta._get_cls(mod_cls, fn_cls, fn_options)
        obj = cls.__new__(cls)
        return obj

    def __getitem__(cls, key):
        assert isinstance(key, (type, tuple, list))

        if isinstance(key, (tuple, list)):
            assert len(key) == 2 \
                   and isinstance(key[0], type) \
                   and isinstance(key[1], dict)
            fn_cls, fn_options = key
        else:
            fn_cls, fn_options = key, {}

        return _MiModuleMeta._get_cls(cls, fn_cls, fn_options)


class NiCategorical(NiModule):
    def __init__(self, name, state, n_inputs=0):
        assert isinstance(state, _base.CategoryNodeState)

        super().__init__(name, state, n_inputs)

        n_classes = state.n_classes
        self.weight = self.sample(
            N(name, 'weight'),
            dist.Dirichlet(torch.ones(n_classes)), )

    def __call__(self, X=None, obs=None):
        result = self.sample(self.name, dist.Categorical(logits=self.weight), obs=obs)
        return result

    @classmethod
    def guide(cls, name, state, n_inputs):
        concentration = cls.param(
            N(name, 'weight', 'concentration'),
            torch.ones([state.n_classes]),
            constraint=dist.Dirichlet.arg_constraints['concentration'], )
        weight = cls.sample(
            N(name, 'weight'),
            dist.Dirichlet(concentration))

        return {N(name, 'weight'): weight}


class NiNormal(NiModule):
    def __init__(self, name, state, n_inputs=0):
        assert isinstance(state, _base.NumericalNodeState)

        super().__init__(name, state, n_inputs)

        self.mean = self.sample(N(name, 'mean'), dist.Normal(state.mean, 1.0))
        self.scale = self.sample(N(name, 'scale'), dist.FoldedDistribution(dist.Normal(state.scale, 1.0)))

    def __call__(self, X=None, obs=None):
        result = self.sample(self.name, dist.Normal(self.mean, self.scale), obs=obs)
        return result

    @classmethod
    def guide(cls, name, state, n_inputs):
        scale_constraint = dist.Normal.arg_constraints['scale']

        mean_mean = cls.param(
            N(name, 'mean', 'mean'),
            torch.tensor(state.mean))
        mean_scale = cls.param(
            N(name, 'mean', 'scale'),
            torch.tensor(1.0),
            constraint=scale_constraint)

        scale_mean = cls.param(
            N(name, 'scale', 'mean'),
            torch.tensor(state.scale),
            constraint=scale_constraint)
        scale_scale = cls.param(
            N(name, 'scale', 'scale'),
            torch.tensor(1.0),
            constraint=scale_constraint)

        mean = cls.sample(
            N(name, 'mean'),
            dist.Normal(mean_mean, mean_scale))
        scale = cls.sample(
            N(name, 'scale'),
            dist.FoldedDistribution(dist.Normal(scale_mean, scale_scale)))

        return {N(name, 'mean'): mean,
                N(name, 'scale'): scale}


class MiCategorical(MiModule, metaclass=_MiModuleMeta):
    """
    Categorical variable with multi-inputs
    """

    def __init__(self, name, state, n_inputs):
        assert isinstance(state, _base.CategoryNodeState)

        super().__init__(name, state, n_inputs)

    def __call__(self, X, obs=None):
        logits = self.process(X)
        result = self.sample(self.name, dist.Categorical(logits=logits), obs=obs)
        return result

    @classmethod
    def guide(cls, name, state, n_inputs):
        return {}


class MiNormal(MiModule, metaclass=_MiModuleMeta):
    """
    Normal distributed numerical variable with multi-inputs
    """

    def __init__(self, name, state, n_inputs):
        assert isinstance(state, _base.NumericalNodeState)

        super().__init__(name, state, n_inputs)
        self.scale = self.sample(
            N(name, 'scale'),
            dist.FoldedDistribution(dist.Normal(state.scale, 1.0)))

    def __call__(self, X, obs=None):
        mean = self.process(X)
        result = self.sample(self.name, dist.Normal(mean.reshape(-1), self.scale), obs=obs)
        return result

    @classmethod
    def guide(cls, name, state, n_inputs):
        scale_mean = cls.param(
            N(name, 'scale', 'mean'),
            torch.tensor(state.scale),
            constraint=dist.Normal.arg_constraints['scale'])
        scale_scale = cls.param(
            N(name, 'scale', 'scale'),
            torch.tensor(1.0),
            constraint=dist.Normal.arg_constraints['scale'])
        scale = cls.sample(
            N(name, 'scale'),
            dist.FoldedDistribution(dist.Normal(scale_mean, scale_scale)))
        return {N(name, 'scale'): scale}


LinearCategorical = MiCategorical[Linear, {'random': 'dirichlet', 'bias': False}]
MLPCategorical = MiCategorical[MLP, {'random': 'dirichlet', 'bias': False}]

LinearNormal = MiNormal[Linear]
MLPNormal = MiNormal[MLP]
