from _typeshed import Self
import copy
import networkx as nx
import numpy as np

from itertools import combinations
from estimator_model.estimation_learner.meta_learner import SLearner, \
    TLearner, XLearner, PropensityScore

np.random.seed(2022)
# 1. add confidence interval


def powerset(iterable):
    s = list(iterable)
    power_set = []
    for i in range(len(s) + 1):
        for j in combinations(s, i):
            power_set.append(set(j))
    return power_set


class CausalModel:
    """Basic object for performing causal inference.

    Attributes
    ----------
    estimator_dic : dictionary
        Keys are estimation methods while values are corresponding objects.
    estimator : EstimationMethod
    causal_graph : causal graph
    data : DataFrame (for now)
        Data is necessary if no causal graph is given.

    Methods
    ----------
    identify(treatment, outcome, identify_method)
        Identify the causal effect expression.
    estimate(X, y, treatment, adjustment_set, target)
        Estimate the causal effect.
    identify_estimate(X, y, treatment, outcome, identify_method, target)
        Combination of identify and estimate.
    discover_graph(data)
        Perform causal discovery.
    is_backdoor_set(set)
        Determine if a given set is a backdoor adjustment set.
    get_backdoor_set(treatment, outcome, adjust)
        Return the backdoor adjustment set for the given treatment and outcome.
    get_backdoor_path(treatment, outcome, graph)
        Return the backdoor path in the graph between treatment and outcome.
    has_collider(path, graph)
        If the path in the current graph has a collider, return True, else
        return False.
    is_connected_backdoor(path, graph)
        Test whether a backdoor path is connected.
    is_frontdoor_set(set)
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
            Describe estimation methods (the first element) and machine
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
            self.data = data
            self.causal_graph = self.discover_graph(self.data)
        else:
            self.causal_graph = causal_graph

    def identifiability(self, treatment, outcome):
        """Determine the identiizbility of treatement and outcome in the current model
        """
        # see Tian and Pearl, 2002a
        pass

    def identify(self, treatment, outcome,
                 identify_method=('backdoor', 'simple')):
        """Identify the causal effect expression.

        Parameters
        ----------
        treatment : str
            name of the treatment
        outcome : str
            name of the outcome
        identify_method : tuple
            two elements where the first one is for the adjustment methods
            and the second is for the returning set style

        Returns
        ----------
        adjustment: list
            the adjustment set
        """
        if identify_method[0] == 'backdoor':
            adjustment_set = self.get_backdoor_set(
                treatment, outcome, adjust=identify_method[1])
        else:
            pass
        return adjustment_set

    def estimate(self, X, y, treatment, adjustment_set, target='ATE'):
        """Estimate the causal effect.

        Parameters
        ----------
        X : DataFrame
        y : DataFrame
            data of the outcome
        treatment : str
            name of the treatment
        adjustment_set : list
        target : str
            the kind of the interested causal effect, including ATE, CATE,
            ITE, and CITE

        Returns
        ----------
        effect : float, optional
        """
        adjustment_set.insert(0, treatment)
        X_adjusted = X[adjustment_set]
        effect = self.estimator.estimate(X_adjusted, y, treatment, target)
        return effect

    def identify_estimate(self, X, y, treatment, outcome,
                          identify_method=('backdoor', 'simple'),
                          target='ATE'):
        """Combination of identifiy and estimate.
        """
        adjustment_set = self.identify(treatment, outcome, identify_method)
        print(f'The corresponding adjustment set is {adjustment_set}')
        if identify_method[1] == 'all':
            adjustment_set = list(adjustment_set[0])
        return self.estimate(X, y, treatment, adjustment_set, target)

    def discover_graph(self, data):
        """Discover the causal graph from data.

        Parameters
        ----------
        data : not sure
        """
        pass

    def is_backdoor_set(self, set, graph=None):
        if graph is None:
            graph = self.causal_graph.DG
        pass

    def get_backdoor_set(self, treatment, outcome, adjust='simple'):
        """Return the backdoor adjustment set for the given treatment and outcome.

        Parameters
        ----------
        treatment : str
            name of the treatment
        outcome : str
            name of the outcome
        adjust : str
            set style of the backdoor set, e.g., simple, minimal and all

        Raises
        ----------
        Exception: raise error if the style is not in simple, minimal or all.

        Returns
        ----------
        backdoor_list : list
        """
        # can I find the adjustment sets by using the adj matrix

        modified_graph = copy.deepcopy(self.causal_graph.DG)
        remove_list = [(treatment, i)
                       for i in modified_graph.successors(treatment)]
        modified_graph.remove_edges_from(remove_list)

        def determine(modified_graph, treatment, outcome):
            if not list(self.causal_graph.DG.predecessors(treatment)):
                return nx.d_separated(modified_graph, {treatment},
                                      {outcome}, {})
            return True

        assert determine(treatment, outcome), \
            'No set can satisfy the backdoor criterion!'

        backdoor_expression = ''
        if adjust == 'simple':
            backdoor_list = list(self.causal_graph.causation[treatment])
            for i in backdoor_list:
                backdoor_expression += f'{i}, '
        else:
            # get all backdoor sets. currently implenmented
            # in a brutal force manner. NEED IMPROVEMENT
            initial_set = (
                set(list(self.causal_graph.causation.keys())) -
                {treatment} - {outcome}
                - set(nx.descendants(self.causal_graph.DG, treatment))
            )
            backdoor_set_list = [
                i for i in powerset(initial_set)
                if nx.d_separated(modified_graph,
                                  {treatment}, {outcome}, i)
            ]

            try:
                backdoor_set_list[0]
            except Exception:
                backdoor_set_list = [[]]
            finally:
                for i in backdoor_set_list[0]:
                    backdoor_expression += f'{i}, '

            if adjust == 'all':
                backdoor_list = backdoor_set_list
            elif adjust == 'minimal':
                backdoor_list = backdoor_set_list[0]
            else:
                raise Exception(
                    'The backdoor set style must be one of simple, all'
                    'and minimal.'
                )

        backdoor_expression = backdoor_expression.strip(', ')

        if backdoor_expression != '':
            print(
                f'The corresponding statistical estimand should be P({outcome}'
                f'|{treatment}, {backdoor_expression})'
            )
        else:
            print(
                f'The corresponding statistical estimand should be P({outcome}'
                f'|{treatment})'
            )
        return backdoor_list

    def get_backdoor_path(self, treatment, outcome, graph=None):
        """Return the backdoor path.

        Parameters
        ----------
        treatment : str
        outcome : str
        graph : CausalGraph

        Returns
        ----------
        A list containing all valid backdoor paths between the treatment and
        outcome in the graph.
        """
        if graph is None:
            graph = self.causal_graph.DG
        return [p for p in nx.all_simple_paths(graph.to_undirected(),
                                               treatment, outcome)
                if len(p) > 2
                and p[1] in graph.predecessors(treatment)]

    def has_collider(self, path, graph=None):
        """If the path in the current graph has a collider, return True, else return False.

        Parameters
        ----------
        path : list
            list containing nodes in the path
        graph : nx.DiGraph, optional

        Returns
        ----------
        Boolean
        """
        if graph is None:
            graph = self.causal_graph.DG
            
        if len(path) > 2:
            for i, node in enumerate(path[1:]):
                if node in graph.successors(path[i-1]):
                    j = i
                    break

            for i, node in enumerate(path[j+1]):
                if node in graph.precedecessors(path[j-1]):
                    return True
        return False

    def is_connected_backdoor(self, path, graph=None):
        """Test whether a backdoor path is connected.

        Parameters
        ----------
        path : list
            a list containing the path
        graph : nx.DiGraph
            the graph in which the criterion is tested

        Returns
        ----------
        Boolean
            True if path is a d-connected backdoor path and False otherwise.
        """
        assert path[1] in graph.precedecessors(path[0]), 'Not a backdoor path.'
        # TODO: improve the implementation

        if path not in self.get_backdoor_path(path[0], path[-1], graph) \
                or self.has_collider(path[1:], graph):
            return False
        return True

    def is_frontdoor_set(self):
        pass

    def get_frontdoor_set(self):
        """See the docstring for get_backdoor_set.
        """
        pass

    def __repr__(self):
        return f'A CausalModel for {self.causal_graph},'\
            f'current support models {self.estimator_dic}'
