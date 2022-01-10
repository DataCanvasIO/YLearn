import copy
import networkx as nx
import numpy as np

from itertools import combinations, chain
from estimator_model import estimation_methods

np.random.seed(2022)


def powerset(iterable):
    """
    https://docs.python.org/3/library/itertools.html#recipes
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r)
                               for r in range(len(s) + 1))


class CausalModel:

    def __init__(self, causal_graph=None, data=None, estimation=None):
        # estimation should be a tuple containing the information like \
        # (machine learning model, estimation method)

        if causal_graph is not None:
            self.graph = causal_graph
        else:
            self.data = data
            self.graph = self.discover_graph(self.data)

        if estimation[1] == 'COM':
            self.estimator = estimation_methods.COM(
                estimation_model=estimation[0])
        elif estimation[1] == 'GroupCOM':
            self.estimator = estimation_methods.GroupCOM(
                estimation_model=estimation[0])
        else:
            pass

    def identify(self, treatment, outcome, identify_method='backdoor'):
        if identify_method == 'backdoor':
            adjustment_set = self.backdoor_set(treatment, outcome)
        else:
            pass
        return adjustment_set

    def estimate(self, X, y, treatment, adjustment_set, target='ATE'):
        adjustment_set.insert(0, treatment)
        X_adjusted = X[adjustment_set]
        effect = self.estimator.estimate(X_adjusted, y, treatment, target)
        return effect

    def identify_estimate(self, X, y, treatment, outcome,
                          identify_method='backdoor', target='ATE'):
        adjustment_set = self.identify(treatment, outcome, identify_method)
        print(f'The corresponding adjustment set is {adjustment_set}')
        return self.estimate(X, y, treatment, adjustment_set, target)

    def discover_graph(self, data):
        pass

    def is_backdoor_set(self, set):
        pass

    def backdoor_set(self, treatment, outcome, adjust='simple'):
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
            backdoor_list = self.graph.causation[treatment]
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
                if nx.d_separated(modified_graph.DG,
                                  {treatment}, {outcome}, {i})
            ]
            for i in backdoor_set_list[0]:
                backdoor_expression += f'{i}, '
            if adjust == 'all':
                backdoor_list = backdoor_set_list
            elif adjust == 'minial':
                backdoor_list = backdoor_set_list[0]
            else:
                pass
        backdoor_expression = backdoor_expression.strip(', ')

        if backdoor_expression != '':
            print(
                f'The corresponding statistical estimand should be P({outcome}'
                f'|{treatment}, {backdoor_expression})')
        else:
            print(
                f'The corresponding statistical estimand should be P({outcome}'
                f'|{treatment})')
        return backdoor_list

    def frontdoor(self):
        pass
