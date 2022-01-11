import copy
import estimator_model
from estimator_model import estimation_methods
import networkx as nx
import numpy as np

from itertools import combinations, chain
from estimator_model.estimation_methods import COM, GroupCOM, XLearner, \
    TLearner, PropensityScore

np.random.seed(2022)


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
    graph : causal graph
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
    is_frontdoor_set(set)
        Determin if a given set is a frontdoor adjustment set.
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
            'COM': COM(ml_model=estimation[0]),
            'GroupCOM': GroupCOM(ml_model=estimation[0]),
            # 'XLearner': XLearner(ml_model=estimation[0]),
            # 'TLearner': TLearner(ml_model=estimation[0]),
            # 'PropensityScore': PropensityScore(ml_model=estimation[0]),
        }

        assert estimation[1] in self.estimator_dic.keys(), \
            'Only support estimation methods in COM, GroupCOM, XLearner,' \
            'TLearner, and PropensityScore.'
        self.estimator = self.estimator_dic[estimation[1]]

        if causal_graph is None:
            assert data is not None, 'Need data to perform causal discovery.'
            self.data = data
            self.graph = self.discover_graph(self.data)
        else:
            self.graph = causal_graph

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
            and the second is for the returning set style.

        Returns
        ----------
        adjustment: list
            the backdoor adjustment set.
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
            describe the kind of the interested causal effect

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

    def is_backdoor_set(self, set):
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
        Exception: raise error when the style is not in simple, minimal and all.

        Returns
        ----------
        backdoor_list : list

        """
        assert (treatment, outcome) in self.graph.edges, \
            f'No direct causal effect between {treatment} and {outcome}' \
            f'exists for the backdoor adjustment.'

        def determine(graph, treatment, outcome):
            return True
        # can I find the adjustment sets by using the adj matrix

        assert determine(
            self.graph.DG, treatment, outcome), \
            'Can not satisfy the backdoor criterion!'

        backdoor_expression = ''
        if adjust == 'simple':
            backdoor_list = list(self.graph.causation[treatment])
            for i in backdoor_list:
                backdoor_expression += f'{i}, '
        else:
            # get all backdoor sets. currently implenmented
            # in a brutal force manner. NEED IMPROVEMENT
            initial_set = (
                set(list(self.graph.causation.keys())) -
                {treatment} - {outcome}
                - set(nx.descendants(self.graph.DG, treatment))
            )
            modified_graph = copy.deepcopy(self.graph.DG)
            modified_graph.remove_edge(treatment, outcome)
            backdoor_set_list = [
                i for i in powerset(initial_set)
                if nx.d_separated(modified_graph,
                                  {treatment}, {outcome}, i)
            ]
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

    def is_frontdoor_set(self):
        pass

    def get_frontdoor_set(self):
        """See the docstring for get_backdoor_set.
        """
        pass
