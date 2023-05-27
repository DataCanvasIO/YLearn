import gzip
from copy import deepcopy
from functools import partial
from io import BytesIO

import numpy as np
import pandas as pd
import pyro
import torch
from pyro import poutine
from pyro.contrib.cevae import TraceCausalEffect_ELBO
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer import autoguide
from pyro.optim import ClippedAdam

from ylearn.utils import logging, to_list, check_fitted, is_notebook, drop_none, set_random_state
from . import _modules, _collectors
from ._base import BObject, CategoryNodeState, BCollectorList
from ._dag import DAG
from ._data import DataLoader

logger = logging.get_logger(__name__)

_DEFAULT_SAMPLE_NUMBER = 100


def _get_pyro_params():
    import pickle
    state = pyro.get_param_store().get_state()
    buf = BytesIO()
    torch.save(state, buf, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    return gzip.compress(buf.getvalue())


def _load_pyro_params(data):
    data = gzip.decompress(data)
    buf = BytesIO(data)
    state = torch.load(buf)
    return state


def with_pyro_ps(fn, check_fitted=True):
    def _with_params(obj, *args, resume_param_state=True, **kwargs):
        if check_fitted:
            assert getattr(obj, '_is_fitted', None) is True, f'{type(obj).__name__} is not fitted.'

        if resume_param_state:
            param_state = getattr(obj, 'pyro_param_state', None)
            # with pyro.get_param_store().scope(param_state):
            #     return fn(obj, *args, **kwargs)

            store = pyro.get_param_store()
            old_state = store.get_state()
            store.clear()
            if param_state is not None:
                store.set_state(param_state)
            try:
                return fn(obj, *args, **kwargs)
            finally:
                store.clear()
                store.set_state(old_state)
        else:
            return fn(obj, *args, **kwargs)

    return _with_params


def with_pyro_ps_(check_fitted=True):
    return partial(with_pyro_ps, check_fitted=check_fitted)


class BayesianModel(BObject):
    NiCategorical = _modules.NiCategorical
    NiNumerical = _modules.NiNormal

    # MiCategorical = _modules.MLPCategorical
    # MiNumerical = _modules.MLPNormal
    #
    MiCategorical = _modules.LinearCategorical
    MiNumerical = _modules.LinearNormal

    def __init__(self, graph, state):
        if not isinstance(graph, DAG):
            logger.info(f'create DAG from {type(graph).__name__}.')
            graph = DAG(graph)

        self.graph = graph
        self.state = state

    @property
    def edges(self):
        return self.graph.get_edges()

    @property
    def nodes(self):
        return self.graph.get_nodes()

    def model(self, data=None, *, n_samples=None, nodes=None, **kwargs):
        if data is None:
            data = {}
            if n_samples is None:
                n_samples = 1
        else:
            n_samples = len(next(iter(data.values())))  # get N from the first data item

        graph = self.graph

        def find_missing_inputs(nodes_):
            missing = set()
            for n in nodes_:
                for p in graph.get_parents(n):
                    if p not in nodes_ and p not in data.keys():
                        missing.add(p)
                        missing.update(find_missing_inputs([p]))
            return missing

        if nodes is not None:
            nodes_to_sample = set(nodes) | find_missing_inputs(nodes)
            nodes_to_sample = graph.sort_nodes(nodes_to_sample, topo_sorted=True)
        else:
            nodes_to_sample = graph.topo_sorted_nodes

        fns = self.node_functions()
        ps = {}

        with pyro.plate('graph', n_samples):
            for node in nodes_to_sample:
                parents = graph.get_parents(node)
                inputs = data.get(f'{node}_inputs')
                if inputs is None:
                    inputs = [ps[p] if p in ps else data[p] for p in parents]
                    # inputs = []
                    # for p in parents:
                    #     if p in ps.keys():
                    #         inputs.append(ps[p])
                    #     elif p in data.keys():
                    #         inputs.append(data[p])
                    #     else:
                    #         raise ValueError('fff')
                fn = fns[node]
                samples = fn(inputs, obs=data.get(node))

                if not isinstance(samples, torch.Tensor):
                    samples = torch.tensor(samples)
                if samples.ndim == 0:
                    # doing intervention
                    samples = samples.repeat(n_samples)
                ps[node] = samples

        if nodes is not None:
            ps = {k: v for k, v in ps.items() if k in nodes}
        return ps

    def node_functions(self, nodes=None):
        if nodes is None:
            nodes = self.graph.nodes

        return {node: self.get_node_function(node) for node in nodes}

    def get_node_function(self, node):
        state = self.state[node]
        parents = self.graph.get_parents(node)
        n_parents = len(parents)
        if isinstance(state, CategoryNodeState):
            if n_parents:
                fn = self.MiCategorical(node, state, n_parents)
            else:
                fn = self.NiCategorical(node, state)
        else:
            if n_parents:
                fn = self.MiNumerical(node, state, n_parents)
            else:
                fn = self.NiNumerical(node, state)

        return fn

    def get_node_function_cls(self, node):
        state = self.state[node]
        parents = self.graph.get_parents(node)
        n_parents = len(parents)
        if isinstance(state, CategoryNodeState):
            if n_parents:
                cls = self.MiCategorical
            else:
                cls = self.NiCategorical
        else:
            if n_parents:
                cls = self.MiNumerical
            else:
                cls = self.NiNumerical

        return cls

    def guide(self, data=None, *, n_samples=None, nodes=None, **kwargs):
        if nodes is None:
            nodes = self.graph.nodes

        graph = self.graph
        result = {}
        for node in nodes:
            cls = self.get_node_function_cls(node)
            state = self.state[node]
            n_inputs = len(graph.get_parents(node))
            result.update(cls.guide(node, state, n_inputs))

        return result

    def __call__(self, data=None, **kwargs):
        return self.model(data, **kwargs)


class BayesianNetwork(BObject):
    # attribute names copied into new stub
    stub_attributes = ('state_', 'model_', 'params_', 'interventions_')

    def __init__(self, graph: DAG):
        if not isinstance(graph, DAG):
            logger.info(f'create DAG from {type(graph).__name__}.')
            graph = DAG(graph)
        self.graph = graph

        # fitted
        self.state_ = None
        self.model_ = None
        self.params_ = None
        self.interventions_ = None
        self._is_fitted = False

    def fit(self, data: pd.DataFrame, *, inplace=True, random_state=None, **kwargs):
        obj = self._fit(data, inplace=inplace, random_state=random_state, **kwargs)
        obj._update_graph_shape()
        obj._is_fitted = True
        return obj

    def _fit(self, data: pd.DataFrame, *, inplace=True, random_state=None, **kwargs):
        raise NotImplementedError()

    def do(self, intervention, data=None, *, n_samples=None,
           blanket='markov', inplace=True, random_state=None, **kwargs):
        raise NotImplementedError()

    @property
    @check_fitted
    def fitted_params(self):
        return self.params_

    @property
    def interventions(self):
        return deepcopy(self.interventions_) if self.interventions_ is not None else {}

    @torch.no_grad()
    def sample(self, data=None, *, n_samples=None, nodes=None, random_state=None):
        model = self.model_
        states = self.state_
        interventions = self.interventions

        if random_state:
            set_random_state(random_state)

        if data is not None:
            if len(interventions) > 0:
                data = data.copy()
                for c, v in interventions.items():
                    data[c] = v
            data_t = DataLoader(states).spread(data)
        else:
            data_t = None
        condition = self.fitted_params
        interventions = self.interventions
        if len(interventions) > 0:
            states = self.state_
            interventions = {k: torch.tensor(states[k].encode(v)) for k, v in interventions.items()}
            condition.update(interventions)
        model = poutine.condition(model, condition)

        samples = model(data=data_t, n_samples=n_samples, nodes=nodes)

        result = {k: states[k].decode(v.numpy()) for k, v in samples.items()}
        result = pd.DataFrame(result, index=data.index if data is not None else None)
        return result

    @check_fitted
    @torch.no_grad()
    def predict(self, data, *, outcome=None, blanket='markov',
                num_samples=_DEFAULT_SAMPLE_NUMBER, proba=False, random_state=None):
        outcome = to_list(outcome, 'outcome')
        states = self.state_
        model = self.model_

        if random_state:
            set_random_state(random_state)

        test_data = DataLoader(states).spread(data)
        for c in outcome:
            if c in test_data.keys():
                test_data.pop(c)

        # if blanket is not None:
        #     nodes = self.graph.get_blanket(targets, kind=blanket, topo_sorted=True)
        #     logger.info(f'blanket size: {len(nodes)}')
        #     model = partial(model, nodes=nodes)
        #     model = poutine.block(model, expose=nodes)

        model = torch.no_grad()(poutine.mask(model, mask=False))
        collector_creator = partial(self._create_collector, proba=proba)
        collectors = BCollectorList(outcome, collector_creator)

        condition = self.fitted_params
        interventions = self.interventions
        if len(interventions) > 0:
            interventions = {k: torch.tensor(states[k].encode(v)) for k, v in interventions.items()}
            condition.update(interventions)

        # sampling
        conditioned_model = poutine.condition(model, condition)
        for i in range(num_samples):
            # if self.infer_ is not None:
            #     posterior = self.infer_.get_samples(num_samples=1)
            #     posterior = {k: v[0] for k, v in posterior.items()}
            # else:
            #     posterior = self.guide_(test_data)
            trace = poutine.trace(conditioned_model).get_trace(test_data)
            # print('G1', trace.nodes['G1']['value'][0])
            collectors(trace)

        # collect result
        result = collectors.to_df()
        logger.info(f'pred:{result.iloc[0].to_dict()}')
        return result

    @with_pyro_ps
    @torch.no_grad()
    def predict_raw(self, data, *, outcome=None, blanket='markov',
                    num_samples=_DEFAULT_SAMPLE_NUMBER, proba=False, random_state=None):
        outcome = to_list(outcome, 'outcome')

        if random_state:
            set_random_state(random_state)

        if self.infer_ is not None:
            mcmc_samples = self.infer_.get_samples(num_samples=num_samples)
            predictive = Predictive(self.model_, posterior_samples=mcmc_samples, num_samples=num_samples,
                                    return_sites=outcome)
        else:
            predictive = Predictive(self.model_, guide=self.guide_, num_samples=num_samples,
                                    return_sites=outcome)

        test_data = DataLoader(self.state_).spread(data)
        samples = predictive(test_data)

        # collect result
        collector_creator = partial(self._create_collector, proba=proba)
        collectors = BCollectorList(outcome, collector_creator)

        for i in range(num_samples):
            values = {k: v[i] for k, v in samples.items()}
            collectors(values)

        result = collectors.to_df()
        return result

    # @with_pyro_ps
    @torch.no_grad()
    def estimate_old(self, data, treatment, outcome, *, treat=None, control=None,
                     num_samples=_DEFAULT_SAMPLE_NUMBER, random_state=None,
                     verbose=None):
        assert data is not None and treatment is not None and outcome is not None

        treatment = to_list(treatment, 'treatment')
        outcome = to_list(outcome, 'outcome')

        if random_state:
            set_random_state(random_state)

        def default_treat_control(node):
            state = self.state_[node]
            if isinstance(state, CategoryNodeState):
                return state.classes[0], state.classes[-1]
            else:
                return state.min, state.max

        if treat is None:
            treat = [default_treat_control(n)[1] for n in treatment]
        else:
            if not isinstance(treat, (list, tuple)):
                treat = [treat, ]
            assert len(treatment) == len(treat), \
                'treatment and treat should have the same number of elements'

        if control is None:
            control = [default_treat_control(n)[0] for n in treatment]
        else:
            if not isinstance(control, (list, tuple)):
                control = [control, ]
            assert len(treatment) == len(control), \
                'treatment and control should have the same number of elements'

        logger.info(f'estimate with control={control} and treat={treat}, '
                    f'treatment={treatment}, outcome={outcome}.')
        states = [self.state_[node] for node in treatment]
        treat_kv = {t: s.encode(v) for t, s, v in zip(treatment, states, treat)}
        control_kv = {t: s.encode(v) for t, s, v in zip(treatment, states, control)}

        model = self.model_
        nodes = self.graph.get_blanket(outcome, kind='upstream', topo_sorted=True)
        cf_model = partial(model.model, nodes=outcome + treatment)
        model = partial(model.model, nodes=nodes)
        model = poutine.block(model, expose=[n for n in nodes if n not in outcome + treatment])
        model = torch.no_grad()(poutine.mask(model, mask=False))
        model = poutine.condition(model, self.fitted_params)

        collector_creator = partial(self._create_collector, proba=True)
        t_collectors = BCollectorList(outcome, collector_creator)
        c_collectors = BCollectorList(outcome, collector_creator)
        model_trace = poutine.trace(model)

        # sampling
        test_data = DataLoader(self.state_).spread(data)
        for i in range(num_samples):
            # with poutine.block(hide=outcome + treatment):
            #         trace_stub = model_trace.get_trace(test_data)
            trace_stub = model_trace.get_trace(test_data)
            with pyro.do(data=treat_kv):
                trace = poutine.trace(poutine.replay(cf_model, trace_stub)).get_trace(test_data)
                t_collectors(trace)
            with pyro.do(data=control_kv):
                trace = poutine.trace(poutine.replay(cf_model, trace_stub)).get_trace(test_data)
                c_collectors(trace)

            # with poutine.trace() as tr, poutine.block(hide=outcome + treatment):
            #     model(test_data)
            # with pyro.do(data=treat_kv):
            #     trace = poutine.trace(poutine.replay(cf_model, tr.trace)).get_trace(test_data)
            #     t_collectors(trace)
            # with pyro.do(data=control_kv):
            #     trace = poutine.trace(poutine.replay(cf_model, tr.trace)).get_trace(test_data)
            #     c_collectors(trace)

            if callable(verbose):
                verbose(i)
            elif verbose and (i + 1) % 20 == 0:
                print('[estimating] %4d' % i)

        # collect result
        ite = t_collectors.to_df() - c_collectors.to_df()
        print(ite.mean(axis=0))
        return ite

    @torch.no_grad()
    def estimate(self, data, treatment, outcome, *, treat=None, control=None,
                 num_samples=_DEFAULT_SAMPLE_NUMBER, random_state=None,
                 verbose=None):
        assert data is not None and treatment is not None and outcome is not None

        treatment = to_list(treatment, 'treatment')
        outcome = to_list(outcome, 'outcome')

        if random_state:
            set_random_state(random_state)

        def default_treat_control(node):
            state = self.state_[node]
            if isinstance(state, CategoryNodeState):
                return state.classes[0], state.classes[-1]
            else:
                return state.min, state.max

        if treat is None:
            treat = [default_treat_control(n)[1] for n in treatment]
        else:
            if not isinstance(treat, (list, tuple)):
                treat = [treat, ]
            assert len(treatment) == len(treat), \
                'treatment and treat should have the same number of elements'

        if control is None:
            control = [default_treat_control(n)[0] for n in treatment]
        else:
            if not isinstance(control, (list, tuple)):
                control = [control, ]
            assert len(treatment) == len(control), \
                'treatment and control should have the same number of elements'

        logger.info(f'estimate with control={control} and treat={treat}, '
                    f'treatment={treatment}, outcome={outcome}.')
        states = [self.state_[node] for node in treatment]
        treat_kv = {t: s.encode(v) for t, s, v in zip(treatment, states, treat)}
        control_kv = {t: s.encode(v) for t, s, v in zip(treatment, states, control)}

        model = self.model_
        nodes = self.graph.get_blanket(outcome, kind='upstream', topo_sorted=True)
        model = partial(model.model, nodes=nodes)
        model = poutine.condition(model, self.fitted_params)

        collector_creator = partial(self._create_collector, proba=True)
        t_collectors = BCollectorList(outcome, collector_creator)
        c_collectors = BCollectorList(outcome, collector_creator)

        # sampling
        test_data = DataLoader(self.state_).spread(data)
        for i in range(num_samples):
            with pyro.condition(data=treat_kv):
                trace = poutine.trace(model).get_trace(test_data)
                t_collectors(trace)
            with pyro.condition(data=control_kv):
                trace = poutine.trace(model).get_trace(test_data)
                c_collectors(trace)

            if callable(verbose):
                verbose(i)
            elif verbose and (i + 1) % 20 == 0:
                print('[estimating] %4d' % i)

        # collect result
        ite = t_collectors.to_df() - c_collectors.to_df()
        print(ite.mean(axis=0))
        return ite

    def estimate_raw(self, data, treatment, outcome, treat=None, control=None, random_state=None):
        obj_treat = self.do({treatment: treat}, data=data.copy(), inplace=False, blanket='markov',
                            random_state=random_state)
        treat_outcome = obj_treat.predict(data, outcome=outcome, proba=True,
                                          random_state=random_state)

        obj_control = self.do({treatment: control}, data=data.copy(), inplace=False, blanket='markov',
                              random_state=random_state)
        control_outcome = obj_control.predict(data, outcome=outcome, proba=True,
                                              random_state=random_state)

        ite = treat_outcome - control_outcome
        return ite

    def _get_stub(self, inplace):
        if inplace:
            return self
        else:
            stub = type(self)(self.graph)
            if self._is_fitted:
                for att in self.stub_attributes:
                    v = getattr(self, att)
                    if v is not None:
                        v = deepcopy(v)
                    setattr(stub, att, v)
            # stub.state_ = self.state_.copy() if self.state_ is not None else None
            # stub.model_ = self.model_
            # stub.guide_ = self.guide_
            # stub.pyro_param_state_ = self.pyro_param_state_ if self.pyro_param_state_ is not None else None
            # stub.infer_ = self.infer_
            # stub.interventions = deepcopy(self.interventions)
            stub._is_fitted = self._is_fitted
            return stub

    def _create_collector(self, name, proba=False):
        state = self.state_[name]
        if isinstance(state, CategoryNodeState):
            return _collectors.CategorySampleCollector(name, state, proba=proba)
        else:
            return _collectors.NumberSampleCollector(name, state)

    def _update_graph_shape(self, ):
        graph = self.graph
        states = self.state_

        for node in graph.get_nodes():
            state = states[node]
            graph.nodes[node]['shape'] = 'box' if isinstance(state, CategoryNodeState) else 'ellipse'

    def plot(self, width=None, height=None, **kwargs):
        assert is_notebook(), f'Plot can only be displayed on notebook.'

        options = drop_none(width=width, height=height, prog='dot')
        options.update(**kwargs)
        self.graph.plot(**options)


class SviBayesianNetwork(BayesianNetwork):
    stub_attributes = BayesianNetwork.stub_attributes + ('guide_', 'pyro_param_state_')

    def __init__(self, graph: DAG):
        super(SviBayesianNetwork, self).__init__(graph)

        # fitted
        self.guide_ = None
        self.pyro_param_state_ = None

    @with_pyro_ps_(check_fitted=False)
    def _fit(self, data: pd.DataFrame, *, epochs=100, lr=0.005, ce_loss=False,
             inplace=True, random_state=None, verbose=None, **kwargs):
        columns = data.columns.tolist()
        nodes = self.graph.get_nodes()
        assert set(columns) >= set(nodes)

        if random_state:
            set_random_state(random_state)

        loader = DataLoader(data=data)
        state = loader.state
        model = BayesianModel(self.graph, state)

        def _is_cat(site):
            n = site['name'].split('_')[0]
            return n in state.keys() and isinstance(state[n], CategoryNodeState)

        gmodel = pyro.poutine.block(model.model, hide=nodes)
        cats = poutine.block(gmodel, expose_fn=_is_cat)
        nums = poutine.block(gmodel, hide_fn=_is_cat)
        guide = autoguide.AutoGuideList(gmodel)
        guide.append(autoguide.AutoNormal(nums))
        guide.append(autoguide.AutoDelta(cats))

        # guide = pyro.poutine.block(model.guide, hide=nodes)
        # guide = autoguide.AutoDelta(pyro.poutine.block(model, hide=nodes))
        # guide = autoguide.AutoNormal(pyro.poutine.block(model, hide=nodes))

        if ce_loss:
            elbo = TraceCausalEffect_ELBO()
        else:
            elbo = Trace_ELBO()

        svi = SVI(model=model,
                  guide=guide,
                  optim=ClippedAdam({"lr": lr, 'clip_norm': 1.0}),
                  loss=elbo,
                  )

        n_weight = len(nodes) * len(data)
        data_t = loader.spread(data)
        for i in range(epochs):
            loss = svi.step(data=data_t) / n_weight
            if callable(verbose):
                verbose(i, loss)
            elif verbose and (i + 1) % 20 == 0:
                print("[iteration %04d] loss: %.4f" % (i + 1, loss))

        with torch.no_grad():
            params = {k: v.detach() for k, v in guide(data_t).items()}
        obj = self._get_stub(inplace)
        obj.state_ = state
        obj.model_ = model
        obj.params_ = params
        obj.guide_ = guide
        obj.pyro_param_state_ = _get_pyro_params()

        return obj

    @with_pyro_ps
    def do(self, intervention, data=None, n_samples=None,
           blanket='markov', epochs=100, lr=0.005, ce_loss=False,
           inplace=True, random_state=None, verbose=1):
        assert isinstance(intervention, dict) and len(intervention) > 0
        if random_state:
            set_random_state(random_state)

        obj = self._get_stub(inplace)

        intervention_ = obj.interventions_ if obj.interventions_ else {}
        intervention_.update(intervention)

        intervention_encoded = {k: self.state_[k].encode(v) for k, v in intervention_.items()}
        model = pyro.do(obj.model_, data=intervention_encoded)
        # guide = pyro.do(obj.guide_, data=intervention_encoded)
        guide = obj.guide_

        # guide = AutoNormal(pyro.poutine.block(BN_model, ))
        if ce_loss:
            elbo = TraceCausalEffect_ELBO()
        else:
            elbo = Trace_ELBO()

        svi = SVI(model=model,
                  guide=guide,
                  optim=ClippedAdam({"lr": lr, 'clip_norm': 1.0}),
                  loss=elbo,
                  )

        if data is not None:
            data = data.copy()
            for c, v in intervention.items():
                data[c] = v
            data_t = DataLoader(self.state_).spread(data)
            n_samples = len(data)
        else:
            data_t = None
            if n_samples is None:
                n_samples = _DEFAULT_SAMPLE_NUMBER

        # nodes = obj.graph.get_nodes()
        blanket_nodes = obj.graph.get_blanket(list(intervention.keys()), kind=blanket, return_self=False)
        # print('blanket:', blanket_nodes)
        n_weight = len(blanket_nodes) * n_samples
        for i in range(epochs):
            loss = svi.step(data=data_t, n_samples=n_samples, nodes=blanket_nodes) / n_weight
            if verbose and (i + 1) % 20 == 0:
                print("[iteration %04d] loss: %.4f" % (i + 1, loss))

        with torch.no_grad():
            params = {k: v.detach() for k, v in guide(data_t).items()}

        # print('>' * 50)
        # from ._zmisc import cmp_params
        # cmp_params(obj.params_, params)
        # print('>' * 50)

        obj.interventions_ = intervention_
        obj.params_ = params
        obj.pyro_param_state_ = _get_pyro_params()

        return obj

    @property
    def pyro_param_state(self):
        if self.pyro_param_state_ is not None:
            return _load_pyro_params(self.pyro_param_state_)
        else:
            return None

    def describe(self, params=True):
        # assert self._is_fitted
        print('>' * 20, type(self).__name__, '<' * 20)
        print('Node  number:', len(self.graph.get_nodes()))
        print('Edge  number:', len(self.graph.get_edges()))

        if self.pyro_param_state_ is not None:
            param_state = self.pyro_param_state['params']
            print('Param number:', np.sum([t.numel() for t in param_state.values()]))
            if params:
                print('-' * 20, 'Parameters', '-' * 20)
                for k, v in param_state.items():
                    if isinstance(v, torch.Tensor):
                        s = f'\t{v.detach().numpy()}'.replace('\n', '\n\t')
                        print(k, ':', f'Tensor{tuple(v.shape)}\n', s)
                    else:
                        print(k, ':', v)

        print('>' * 20, 'DONE', '<' * 20)
