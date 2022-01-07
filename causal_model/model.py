import networkx as nx

from estimator_model import estimation_methods


class CausalModel:

    def __init__(self, causal_graph=None, data=None, estimation=None):
        # estimation should be a tuple containing the information like (machine learning model, estimation method)

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

    def estimate(self, X, y, treatment, adjustment_set,
                 target='ATE'):
        adjustment_set.insert(0, treatment)
        X_adjusted = X[adjustment_set]
        effect = self.estimator.estimate(
            X_adjusted, y, treatment, target)
        return effect

    def identify_estimate(self, X, y, treatment, outcome, identify_method='backdoor',
                          target='ATE'):
        adjustment_set = self.identify(treatment, outcome, identify_method)
        print(f'The corresponding adjustment set is {adjustment_set}')
        return self.estimate(X, y, treatment, adjustment_set, target)

    def discover_graph(self, data):
        pass

    def backdoor_set(self, treatment, outcome, minimal=False):
        def determine(treatment, outcome):
            return True
        # can I find the adjustment sets by using the adj matrix

        assert determine(
            treatment, outcome), 'Not satisfy the backdoor criterion!'

        backdoor_expression = ''
        if not minimal:
            backdoor_list = list(
                set(self.graph.causation[treatment] +
                    self.graph.causation[outcome])
            )
            backdoor_list = [i for i in backdoor_list if i !=
                             treatment and i != outcome]
            for i in backdoor_list:
                backdoor_expression += f'{i}, '
        else:
            pass
        backdoor_expression = backdoor_expression.strip(', ')

        if backdoor_expression != '':
            print(
                f'The corresponding statistical estimand should be P({outcome}|{treatment}, {backdoor_expression})')
        else:
            print(
                f'The corresponding statistical estimand should be P({outcome}|{treatment})')
        return backdoor_list

    def frontdoor(self):
        pass
