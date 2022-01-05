import networkx as nx


class CausalModel:

    def __init__(self, causal_graph=None, data=None):
        if causal_graph is not None:
            self.graph = causal_graph
        else:
            self.graph = self.discover(data)

    def identify(self, treatment, outcome, identify_method='backdoor'):
        if identify_method == 'backdoor':
            set = self.backdoor_set(treatment, outcome)
            return set

    def estimate(self, data, treatment, outcome, estimator, target='ATE'):
        pass

    def identify_estimate(self, data, treatment, outcome, identify_method,
                          estimator, target='ATE'):
        pass

    def discover(self, data):
        pass

    def backdoor_set(self, treatment, outcome, minimal=False):
        def determine(treatment, outcome):
            return True
        # can I find the adjustment sets by using the adj matrix

        identifibility = determine(treatment, outcome)
        assert identifibility, 'Not satisfy the backdoor criterion!'

        backdoor_list = []
        backdoor_expression = ''
        if not minimal:
            self.graph.DG.remove_edge(treatment, outcome)
            for i in self.graph.edges:
                if i[1] == treatment and \
                        nx.d_separated(self.graph.DG, {treatment}, {outcome},
                                       {i[0]}):
                    backdoor_list.append(i[0])
                    backdoor_expression += f'{i[0]}, '
        else:
            pass
        backdoor_expression = backdoor_expression.strip(', ')

        print(
            f'The corresponding statistical estimand should be P({outcome}|{treatment}, {backdoor_expression})')
        return backdoor_list

    def frontdoor(self):
        pass
