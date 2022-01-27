import networkx as nx
import numpy as np

from causal_model.prob import Prob
from itertools import combinations
from estimator_model.estimation_learner.meta_learner import SLearner, \
    TLearner, XLearner, PropensityScore

np.random.seed(2022)


def powerset(iterable):
    """
    Return the power set of the iterable.

    Parameters
    ----------
    iterable : container
        Can be a set or a list.

    Returns
    ----------
    list
        The list of power set.
    """
    s = list(iterable)
    power_set = []
    for i in range(len(s) + 1):
        for j in combinations(s, i):
            power_set.append(set(j))
    return power_set


class IdentificationError(Exception):
    def __init__(self, info):
        super().__init__(self)
        self.info = info

    def __str__(self):
        return self.info


class CausalModel:
    """
    Basic object for performing causal inference.

    Attributes
    ----------
    estimator_dic : dictionary
        Keys are estimation methods while values are corresponding objects.
    estimator : estimation_learner
    causal_graph : CausalGraph #TODO: support CausalStructuralModel

    Methods
    ----------
    id(y, x, prob=None, graph=None)
        Identify the causal quantity P(y|do(x)) in the graph if identifiable
        else return False, where y can be a set of different outcomes and x
        can be a set of different treatments. #TODO: be careful about that
        currently we treat a random variable equally as its value, i.e., we
        do not discern P(Y|do(X)) and P(Y=y|do(X=x)). I don't know if this
        matters.
    identify(treatment, outcome, identify_method=None)
        Identify the causal effect of treatment on outocme expression.
    estimate(x, y, treatment, adjustment_set, target='ATE')
        Estimate the causal effect of treatment on outcome (y).
    identify_estimate(
        x, y, treatment, outcome, identify_method=None, target='ATE'
    )
        Combination of identify and estimate.
    discover_graph(data)
        Perform causal discovery over data.
    is_valid_backdoor_set(set, treatment, outcome, graph=None)
        Determine if a given set is a valid backdoor adjustment set in the
        graph.
    get_backdoor_set(treatment, outcome, adjust='simple')
        Return backdoor adjustment sets for the given treatment and outcome
        in the style stored in adjust (simple, minimal, or all).
    get_backdoor_path(treatment, outcome, graph=None)
        Return all backdoor paths in the graph between treatment and outcome.
    has_collider(path, graph=None, backdoor_path=True)
        If the path in the current graph has a collider, return True, else
        return False.
    is_connected_backdoor_path(path, graph=None)
        Test whether a backdoor path is connected in the current graph, where
        path[0] is the start of the path and path[-1] is the end.
    is_frontdoor_set(set, treatment, outcome)
        Determine if a given set is a frontdoor adjustment set.
    get_frontdoor_set(treatment, outcome, adjust)
        Return the frontdoor adjustment set for given treatment and outcome.
    """

    def __init__(self, causal_graph=None, data=None, estimation=None):
        """
        Parameters
        ----------
        causal_graph : CausalGraph
        data : DataFrame (for now)
        estimation : tuple of 2 elements
            describe estimation methods (the first element) and machine
            learning models (the second element) used for estimation
        """

        self.estimator_dic = {
            'S-Learner': SLearner(ml_model=estimation[0]),
            'T-Learner': TLearner(ml_model=estimation[0]),
            # 'XLearner': XLearner(ml_model=estimation[0]),
            # 'TLearner': TLearner(ml_model=estimation[0]),
            # 'PropensityScore': PropensityScore(ml_model=estimation[0]),
        }

        assert estimation[1] in self.estimator_dic.keys(), \
            f'Only support estimation methods in {self.estimator_dic.keys()}'
        self.estimator = self.estimator_dic[estimation[1]]

        if causal_graph is None:
            assert data is not None, 'Need data to perform causal discovery.'
            self.causal_graph = self.discover_graph(data)
        else:
            self.causal_graph = causal_graph

    def id(self, y, x, prob=None, graph=None):
        """
        Identify the causal quantity P(y|do(x)) if identifiable else return False.
        see Shpitser and Pearl (2006b)
        (https://ftp.cs.ucla.edu/pub/stat_ser/r327.pdf) for reference.

        Parameters
        ----------
        y : set
            Set of str type outcomes.
        x : set
            Set of str type treatments.
        prob : Prob
            Probability distribution encoded in the graph.
        graph : CausalGraph or CausalStructuralModel

        Returns
        ----------
        Prob if identifiable

        Raises
        ----------
        IdentificationError if not identifiable
        """
        # TODO: need to be careful about the fact that set is not ordered,
        # be careful about the usage of list and set in the current
        # implementation
        if graph is None:
            graph = self.causal_graph

        if prob is None:
            prob = graph.prob

        v = graph.causation.keys()
        v_topo = list(graph.topo_order)

        # 1
        if not x:
            if (prob.divisor is not None) or (prob.product is not None):
                prob.marginal = v.difference(y).union(prob.marginal)
            else:
                prob.variables = y
                return prob

        # 2
        ancestor = graph.ancestors(y)
        if v.difference(ancestor) != set():
            an_graph = graph.build_sub_graph(ancestor, new=True)
            if (prob.divisor is not None) or (prob.product is not None):
                prob.marginal = v.difference(ancestor).union(prob.marginal)
            else:
                prob.variables = ancestor
            return self.id(y, x.intersection(ancestor), prob, an_graph)

        # 3
        w = v.difference(x).difference(
            graph.remove_incoming_edges(x, new=True).ancestors(y)
        )
        if w:
            return self.id(y, x.union(w), prob, graph)

        # 4
        c = graph.remove_nodes(x, new=True).c_components
        if len(c) > 1:
            product_expressioin = set()
            for subset in c:
                product_expressioin.add(
                    self.id(subset, v.difference(subset), prob, graph)
                )
            return Prob(
                marginal=v.difference(y.union(x)), product=product_expressioin
            )
        else:
            s = c.pop()
            cg = set(graph.c_componets)
            c_ = cg.pop()
            # 5
            if (c_ == v) and (len(cg) == 1):
                raise IdentificationError(
                    'The causal quantity is not identifiable in the'
                    'current graph.'
                )
            # 6
            elif s.intersection(c_) == s:
                product_expressioin = set()
                for element in s:
                    product_expressioin.add(
                        Prob(variables=set(element),
                             conditional=set(v_topo[:v_topo.index(element)]))
                    )
                return Prob(
                    marginal=s.difference(y), product=product_expressioin
                )

            # 7
            else:
                # TODO: not clear whether directly replacing a random variable
                # with one of its value matters in this line
                for subset in cg:
                    if s.intersection(subset) == s:
                        product_expressioin = set()
                        for element in subset:
                            product_expressioin.add(
                                Prob(variables=set(element),
                                     conditional=set(v_topo[:v_topo.index(
                                         element)]))
                            )
                        sub_prob = Prob(product=product_expressioin)
                        sub_graph = graph.build_sub_graph(subset)
                        return self.id(
                            y, x.intersection(subset), sub_prob, sub_graph
                        )

    def identify(self, treatment, outcome, identify_method=None):
        """
        Identify the causal effect expression.

        Parameters
        ----------
        treatment : str or set
            Set of names of the treatments.
        outcome : str or set
            Set of names of the outcomes.
        identify_method : tuple
            Two elements where the first one is for the adjustment methods
            and the second is for the returned set style.
            adjustment methods include backdoor, frontdoor, general_adjust, and
            default while returned set styles include simple, minimal,
            all, and default.

        Returns
        ----------
        adjustment
            Prob if identify_method is ('default', 'default'), ortherwise
            return (list (the adjustment set), Prob).
        # TODO: should support more general identification,
        # e.g., probability distribution
        """
        if identify_method is None \
                or identify_method == ('default', 'default'):
            return self.id(outcome, treatment)
        elif identify_method[0] == 'backdoor':
            adjustment = self.get_backdoor_set(
                treatment, outcome, adjust=identify_method[1]
            )
        elif identify_method[0] == 'frontdoor':
            adjustment = self.get_frontdoor_set(treatment, outcome)
        else:
            adjustment = None
        return adjustment

    def estimate(self, x, y, treatment, adjustment_set, target='ATE'):
        """
        Estimate the causal effect.

        Parameters
        ----------
        x : DataFrame
        y : DataFrame
            Data of the outcome.
        treatment : str
            Name of the treatment.
        adjustment_set : list
        target : str
            The kind of the interested causal effect, including ATE, CATE,
            ITE, and CITE.

        Returns
        ----------
        effect : float, optional
        """
        adjustment_set.insert(0, treatment)
        x_adjusted = x[adjustment_set]
        effect = self.estimator.estimate(x_adjusted, y, treatment, target)
        return effect

    def identify_estimate(self, x, y, treatment, outcome,
                          identify_method=('backdoor', 'simple'),
                          target='ATE'):
        """
        Combination of identifiy and estimate.

        Parameters
        ----------
        x : DataFrame
            Data of the treatment.
        y : DataFrame
            Data of the outcome.
        treatment : str
            Name of the treatment.
        outcome : str
            Name of the outcome.
        identify_method : tuple
            Refer to docstring of identify().
        target : str
            The kind of the interested causal effect, including ATE, CATE,
            ITE, and CITE.

        Returns
        ----------
        effect : float, optional
        """
        adjustment_set = self.identify(treatment, outcome, identify_method)
        print(f'The corresponding adjustment set is {adjustment_set}')
        if identify_method[1] == 'all':
            adjustment_set = list(adjustment_set[0])
        return self.estimate(x, y, treatment, adjustment_set, target)

    def discover_graph(self, data):
        """
        Discover the causal graph from data.

        Parameters
        ----------
        data : not sure
        """
        pass

    def is_valid_backdoor_set(self, set_, treatment, outcome, graph=None):
        """
        Determine if a given set is a valid backdoor adjustment set for
        causal effect of treatments on the outcomes.

        Parameters
        ----------
        set : set
            The adjustment set.
        treatment : str
        outcome : str
        graph : CausalGraph
            If None, use self.causal_graph. Defaults to None.

        Returns
        ----------
        Bool
            True if the given set is a valid backdoor adjustment set.
        """
        if graph is None:
            graph = self.causal_graph.dag
        return (set_ in self.get_backdoor_set(treatment, outcome, graph))

    def get_backdoor_set(self, treatment, outcome, adjust='simple'):
        """
        Return the backdoor adjustment set for the given treatment and outcome.

        Parameters
        ----------
        treatment : str or set
            Name of the treatment.
        outcome : str or set
            Name of the outcome.
        adjust : str
            Set style of the backdoor set
                simple: directly return the parent set of treatment
                minimal: return the minimal backdoor adjustment set
                all: return all valid backdoor adjustment set.

        Raises
        ----------
        Exception : IdentificationError
            Raise error if the style is not in simple, minimal or all or no
            set can satisfy the backdoor criterion.

        Returns
        ----------
        tuple
            The first element is the adjustment list, the second is encoded
            Prob.
        """
        # TODO: can I find the adjustment sets by using the adj matrix
        # TODO: improve the implementation
        # TODO: support set of treatments and outcomes
        # modified_graph = self.causal_graph.dag.copy()
        # remove_list = [(treatment, i)
        #                for i in modified_graph.successors(treatment)]
        # modified_graph.remove_edges_from(remove_list)
        modified_graph = self.causal_graph.remove_incoming_edges(
            set(treatment), new=True
        )

        def determine(modified_graph, treatment, outcome):
            if not list(
                self.causal_graph.observed_graph.predecessors(treatment)
            ):
                return nx.d_separated(modified_graph, {treatment},
                                      {outcome}, set())
            return True

        assert determine(treatment, outcome), \
            'No set can satisfy the backdoor criterion!'

        # backdoor_expression = ''
        if adjust == 'simple':
            backdoor_list = list(self.causal_graph.causation[treatment])
            # for i in backdoor_list:
            #     backdoor_expression += f'{i}, '
        else:
            # get all backdoor sets. currently implenmented
            # in a brutal force manner. NEED IMPROVEMENT
            initial_set = (
                set(list(self.causal_graph.causation.keys())) -
                {treatment} - {outcome}
                - set(nx.descendants(self.causal_graph.dag, treatment))
            )
            backdoor_set_list = [
                i for i in powerset(initial_set)
                if nx.d_separated(modified_graph,
                                  {treatment}, {outcome}, i)
            ]

            # try:
            #     backdoor_set_list[0]
            # except Exception:
            #     backdoor_set_list = [[]]
            # finally:
            #     for i in backdoor_set_list[0]:
            #         backdoor_expression += f'{i}, '

            if adjust == 'all':
                backdoor_list = backdoor_set_list
            elif adjust == 'minimal':
                backdoor_list = backdoor_set_list[0]
            else:
                raise IdentificationError(
                    'The backdoor set style must be one of simple, all'
                    'and minimal.'
                )

        adset = set(backdoor_list)
        product_expression = {
            Prob(variables=outcome,
                 conditional=treatment.update(adset)),
            Prob(variables=adset)
        }
        prob = Prob(marginal=adset, product=product_expression)
        print(
            f'The corresponding statistical estimand should be {prob.parse()})'
        )
        # backdoor_expression = backdoor_expression.strip(', ')
        # if backdoor_expression != '':
        #     print(
        #         f'The corresponding statistical estimand should be P({outcome}'
        #         f'|{treatment}, {backdoor_expression})'
        #     )
        # else:
        #     print(
        #         f'The corresponding statistical estimand should be P({outcome}'
        #         f'|{treatment})'
        #     )
        return (backdoor_list, prob)

    def get_backdoor_path(self, treatment, outcome, graph=None):
        """
        Return all backdoor path connecting treatment and outcome.

        Parameters
        ----------
        treatment : str
        outcome : str
        graph : nx.DiGraph

        Returns
        ----------
        list
            A list containing all valid backdoor paths between the treatment and
            outcome in the graph.
        """
        if graph is None:
            graph = self.causal_graph.observed_graph
        return [
            p for p in nx.all_simple_paths(graph.to_undirected(),
                                           treatment, outcome)
            if len(p) > 2
            and p[1] in graph.predecessors(treatment)
        ]

    def has_collider(self, path, graph=None, backdoor_path=True):
        """
        If the path in the current graph has a collider, return True, else
        return False.

        Parameters
        ----------
        path : list
            A list containing nodes in the path.
        graph : nx.DiGraph, optional

        Returns
        ----------
        Boolean
            True if the path has a collider.
        """
        # TODO: improve the implementation.
        if len(path) > 2:
            if graph is None:
                graph = self.causal_graph.dag

            # assert (
            #     path in list(nx.all_simple_paths(graph.to_undirected,
            #                                      path[0], path[-1]))
            # ), "Not a valid path."

            assert(nx.is_path(graph.to_undirected, path)), "Not a valid path."

            if backdoor_path:
                for i, node in enumerate(path[1:]):
                    if node in graph.successors(path[i-1]):
                        j = i
                        break
                for i, node in enumerate(path[j+1]):
                    if node in graph.precedecessors(path[j-1]):
                        return True
            else:
                for i, node in enumerate(path[1:]):
                    if node in graph.precedecessors(path[i-1]):
                        j = i
                        break
                for i, node in enumerate(path[j+1]):
                    if node in graph.successors(path[j-1]):
                        return True
        return False

    def is_connected_backdoor_path(self, path, graph=None):
        """
        Test whether a backdoor path is connected.

        Parameters
        ----------
        path : list
            A list containing the path.
        graph : nx.DiGraph
            The graph in which the criterion is tested.

        Returns
        ----------
        Boolean
            True if path is a d-connected backdoor path and False otherwise.
        """
        if graph is None:
            graph = self.causal_graph.dag

        assert path[1] in graph.precedecessors(path[0]), 'Not a backdoor path.'
        # TODO: improve the implementation

        if self.has_collider(path[1:], graph) or \
                path not in self.get_backdoor_path(path[0], path[-1], graph):
            return False
        return True

    def is_frontdoor_set(self, set_, treatment, outcome):
        """
        True is the given set is a valid frontdoor adjustment set.

        Parameters
        ----------
        set_ : set
        treatement : str or set
        outcome : str or set

        Returns
        ----------
        Bool
            True if the given set is a valid frontdoor adjustment set for
            corresponding treatemtns and outcomes.
        """
        pass

    def get_frontdoor_set(self, treatment, outcome):
        """See the docstring for get_backdoor_set.
        """
        pass

    def __repr__(self):
        return f'A CausalModel for {self.causal_graph},'\
            f'which currently supports models {self.estimator_dic}'
