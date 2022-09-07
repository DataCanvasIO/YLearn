from copy import deepcopy
from collections import defaultdict
from turtle import clone

import networkx as nx
import numpy as np

from sklearn.linear_model import LinearRegression as LR

from ylearn.utils._common import check_cols
from ylearn.estimator_model.double_ml import DoubleML
from .utils import (
    check_nodes,
    check_ancestors_chain,
    powerset,
    IdentificationError,
    remove_ingo_edges,
    descendents_of_iter,
)
from .prob import Prob

np.random.seed(2022)


class CausalModel:
    """
    Basic object for performing causal inference.

    Attributes
    ----------
    data : pandas.DataFrame
        Data used for discovering causal graph and training estimator model.

    causal_graph : CausalGraph #TODO: support CausalStructuralModel

    treatment : list of str, optional
        Names of treatments in the current identification problem.

    outcome : list of str, optional
        Names of outcomes in the current identification problem.

    ava_nodes : dictkeys
        All observed nodes of the CausalGraph

    cached_result : dict
        Results of previous identification results.

    _adjustment_set : list of set
        All possible backdoor adjustment set

    Methods
    ----------
    id(y, x, prob=None, graph=None)
        Identify the causal quantity P(y|do(x)) in the graph if identifiable
        else return False, where y can be a set of different outcomes and x
        can be a set of different treatments. #TODO: be careful about that
        currently we treat a random variable equally as its value, i.e., we
        do not discern P(Y|do(X)) and P(Y=y|do(X=x)).

    identify(treatment, outcome, identify_method='auto')
        Identify the causal effect of treatment on outocme expression.

    estimate(estimator_model, data, adjustment, covariate, quantity=None)
        Estimate the causal effect of treatment on outcome (y).

    get_iv(treatment, outcome)
        Find the instrumental variables for the causal effect of the
        treatment on the outcome.

    is_iv(treatment, outcome, set_)
        Determine whether a given set_ is a valid instrumental variable set.

    discover_graph(data)
        Perform causal discovery over data.

    is_valid_backdoor_set(set, treatment, outcome)
        Determine if a given set is a valid backdoor adjustment set for causal
        effects of treatments on outcomes.

    get_backdoor_set(treatment, outcome, adjust='simple')
        Return backdoor adjustment sets for the given treatment and outcome
        in the style stored in adjust (simple, minimal, or all).

    get_backdoor_path(treatment, outcome)
        Return all backdoor paths in the graph between treatment and outcome.

    has_collider(path, backdoor_path=True)
        If the path in the current graph has a collider, return True, else
        return False.

    is_connected_backdoor_path(path)
        Test whether a backdoor path is connected in the current graph, where
        path[0] is the start of the path and path[-1] is the end.

    is_frontdoor_set(set, treatment, outcome)
        Determine if a given set is a frontdoor adjustment set.

    get_frontdoor_set(treatment, outcome, adjust)
        Return the frontdoor adjustment set for given treatment and outcome.

    estimate_hidden_cofounder()
        Estimation of hidden cofounders.
    """

    def __init__(self, causal_graph=None, data=None, **kwargs):
        """
        Parameters
        ----------
        causal_graph : CausalGraph
            An instance of CausalGraph which encodes the causal structures.

        data : DataFrame (for now)
        """
        self.data = data
        self._adjustment_set = None
        self.treatment = None
        self.outcome = None
        self.cached_result = defaultdict(list)

        self.causal_graph = (
            causal_graph if causal_graph is not None else self.discover_graph(data)
        )

        # self.ava_nodes = self.causal_graph.causation.keys()

    def id(self, y, x, prob=None, graph=None):
        """Identify the causal quantity P(y|do(x)) if identifiable else return
        False. See Shpitser and Pearl (2006b)
        (https://ftp.cs.ucla.edu/pub/stat_ser/r327.pdf) for reference.
        Note that here we only consider semi-Markovian causal model, where
        each unobserved variable is a parent of exactly two nodes. This is
        because any causal model with unobserved variables can be converted
        to a semi-Markovian causal model encoding the same set of conditional
        independences (Verma, 1993).

        Parameters
        ----------
        y : set of str
            Set of names of outcomes.

        x : set of str
            Set of names of treatments.

        prob : Prob
            Probability distribution encoded in the graph.

        graph : CausalGraph

        Returns
        ----------
        Prob if identifiable.

        Raises
        ----------
        IdentificationError if not identifiable.
        """
        # TODO: need to be careful about the fact that sets are not ordered,
        # be careful about the usage of list and set in the current
        # implementation
        if graph is None:
            graph = self.causal_graph

        if prob is None:
            prob = graph.prob

        v = set(graph.causation.keys())
        v_topo = list(graph.topo_order)

        y, x = set(y), set(x)

        # 1
        if not x:
            if prob.divisor or prob.product:
                prob.marginal = v.difference(y).union(prob.marginal)
            else:
                # If the prob is not built with products, then
                # simply replace the variables with y
                prob.variables = y
            return prob

        # 2
        ancestor = graph.ancestors(y)
        prob_ = deepcopy(prob)
        if v.difference(ancestor) != set():
            an_graph = graph.build_sub_graph(ancestor)
            if prob_.divisor or prob_.product:
                prob_.marginal = v.difference(ancestor).union(prob_.marginal)
            else:
                prob_.variables = ancestor
            return self.id(y, x.intersection(ancestor), prob_, an_graph)

        # 3
        w = v.difference(x).difference(
            graph.remove_incoming_edges(x, new=True).ancestors(y)
        )
        if w:
            return self.id(y, x.union(w), prob, graph)

        # 4
        c = list(graph.remove_nodes(x, new=True).c_components)
        if len(c) > 1:
            product_expressioin = set()
            for subset in c:
                product_expressioin.add(
                    self.id(subset, v.difference(subset), prob, graph)
                )
            return Prob(marginal=v.difference(y.union(x)), product=product_expressioin)
        else:
            s = c.pop()
            cg = list(graph.c_components)
            # 5
            if cg[0] == set(graph.dag.nodes):
                raise IdentificationError(
                    "The causal effect is not identifiable in the" "current graph."
                )
            # 6
            elif s in cg:
                product_expression = set()
                for element in s:
                    product_expression.add(
                        Prob(
                            variables={element},
                            conditional=set(v_topo[: v_topo.index(element)]),
                        )
                    )
                return Prob(marginal=s.difference(y), product=product_expression)

            # 7
            else:
                # TODO: not clear whether directly replacing a random variable
                # with one of its value matters in this line
                for subset in cg:
                    if s.intersection(subset) == s:
                        product_expressioin = set()
                        for element in subset:
                            product_expressioin.add(
                                Prob(
                                    variables={element},
                                    conditional=set(v_topo[: v_topo.index(element)]),
                                )
                            )
                        sub_prob = Prob(product=product_expressioin)
                        sub_graph = graph.build_sub_graph(subset)
                        return self.id(y, x.intersection(subset), sub_prob, sub_graph)

    def identify(self, treatment, outcome, identify_method="auto"):
        """Identify the causal effect expression. Identification is an operation that
        converts any causal effect quantity, e.g., quantities with the do operator, into
        the corresponding statistical quantity such that it is then possible
        to estimate the causal effect in some given data. However, note that not all
        causal quantities are identifiable, in which case an IdentificationError
        will be raised.

        Parameters
        ----------
        treatment : set or list of str, optional
            Set of names of treatments.

        outcome : set or list of str, optional
            Set of names of outcomes.

        identify_method : tuple of str or str, optional. Default to 'auto'.
            If the passed value is a tuple or list, then it should have two
            elements where the first one is for the identification methods
            and the second is for the returned set style.

            Available options:
            'auto' : Perform identification with all possible methods
            'general': The general identification method, see id()
            ('backdoor', 'simple'): Return the set of all direct confounders of
                                    both treatments and outcomes as a backdoor
                                    adjustment set.
            ('backdoor', 'minimal'): Return all possible backdoor adjustment sets with
                                     minial number of elements.
            ('backdoor', 'all'): Return all possible backdoor adjustment sets.
            ('frontdoor', 'simple'): Return all possible frontdoor adjustment sets with
                                      minial number of elements.
            ('frontdoor', 'minimal'): Return all possible frontdoor adjustment sets with
                                      minial number of elements.
            ('frontdoor', 'all'): Return all possible frontdoor adjustment sets.

        Returns
        ----------
        dict
            Keys of the dict are identify methods while the values are the
            corresponding results.

        # TODO: support finding the minimal general adjustment set in linear
        # time
        """
        assert treatment and outcome, "Please provide names of treatments and outcomes."

        # check nodes avaliable in the graph
        ava_nodes = self.causal_graph.causation.keys()
        check_nodes(ava_nodes, treatment, outcome)

        self.treatment = treatment
        self.outcome = outcome

        if identify_method == "auto":
            result_dict = {
                "ID": self.id(outcome, treatment),
                "backdoor": self.get_backdoor_set(treatment, outcome, "simple"),
                "frontdoor": self.get_frontdoor_set(treatment, outcome, "simple"),
            }
            self.cached_result[f"Treatment: {treatment}, Outcome: {outcome}"].append(
                result_dict
            )
            return result_dict

        if identify_method == "general":
            result = self.id(outcome, treatment)
            self.cached_result[f"Treatment: {treatment}, Outcome: {outcome}"].append(
                {identify_method: result}
            )
            return {"ID": result}

        if identify_method[0] == "backdoor":
            adjustment = self.get_backdoor_set(
                treatment, outcome, adjust=identify_method[1]
            )
            self.cached_result[f"Treatment: {treatment}, Outcome: {outcome}"].append(
                {identify_method[0]: adjustment}
            )
        elif identify_method[0] == "frontdoor":
            adjustment = self.get_frontdoor_set(
                treatment, outcome, adjust=identify_method[1]
            )
            self.cached_result[f"Treatment: {treatment}, Outcome: {outcome}"].append(
                {identify_method[0]: adjustment}
            )
        else:
            raise IdentificationError(
                "Support backdoor adjustment, frontdoor adjustment, and general"
                f"identification, but was given {identify_method[1]}"
            )

        return {f"{identify_method[0]}": adjustment}

    def estimate(
        self,
        estimator_model,
        data=None,
        *,
        treatment=None,
        outcome=None,
        adjustment=None,
        covariate=None,
        quantity=None,
        **kwargs,
    ):
        """Estimate the identified causal effect in a new dataset.

        Parameters
        ----------
        estimator_model : EstimatorModel
            Any suitable estimator models implemented in the EstimatorModel can
            be applied here.

        data : pandas.Dataframe, optional. Default to None
            The data set for causal effect to be estimated. If None, use the data
            which is used for discovering causal graph.

        treatment : set or list, optional. Default to None
            Names of the treatment. If None, the treatment used for backdoor adjustment
            will be taken as the treatment.

        outcome : set or list, optional. Default to None
            Names of the outcome. If None, the treatment used for backdoor adjustment
            will be taken as the outcome.

        adjustment : set or list, optional. Default to None
            Names of the adjustment set. If None, the ajustment set is given by
            the simplest backdoor set found by CausalModel.

        covariate : set or list, optional. Default to None
            Names of covariate set. Ignored if set as None.

        quantity : str, optional. Default to None
            The interested quantity when evaluating causal effects.

        Returns
        -------
        ndarray or float
            The estimated causal effect in data.
        """
        data = self.data if data is None else data
        check_cols(data, treatment, outcome, adjustment, covariate)

        ava_nodes = self.causal_graph.causation.keys()
        check_nodes(ava_nodes, adjustment, covariate)

        # assert self.outcome is not None
        # assert self.treatment is not None

        outcome = self.outcome if outcome is None else outcome
        treatment = self.treatment if treatment is None else treatment

        # make sure the adjustment and the covariate are valid backdoor adjustment sets
        if adjustment is None and covariate is None:
            if self._adjustment_set is None:
                adjustment = self.get_backdoor_set(
                    self.treatment, self.outcome, "simple"
                )[0]
            else:
                adjustment = self._adjustment_set[0]
        else:
            if adjustment is not None and covariate is not None:
                temp_set = set(adjustment).union(set(covariate))
            else:
                temp_set = [set(x) for x in [adjustment, covariate] if x is not None]
                temp_set = temp_set.pop()

            assert self.is_valid_backdoor_set(
                temp_set, self.treatment, self.outcome
            ), "The adjustment set should be a valid backdoor adjustment set,"
            f"but was given {temp_set}."

        # fit the estimator_model if it is not fitted
        if not estimator_model._is_fitted:
            estimator_model.fit(
                data=data,
                outcome=outcome,
                treatment=treatment,
                covariate=covariate,
                adjustment=adjustment,
                **kwargs,
            )

        effect = estimator_model.estimate(data=data, quantity=quantity)

        return effect

    def identify_estimate(
        self,
        data,
        outcome,
        treatment,
        estimator_model=None,
        quantity=None,
        identify_method="auto",
        **kwargs,
    ):
        """Combination of the identifiy method and the estimate method. However,
        since current implemented estimator models assume (conditionally)
        unconfoundness automatically (except for methods related to iv), we may
        only consider using backdoor set adjustment to fullfill the unconfoundness
        condition.

        Parameters
        ----------
        data : pandas.DataFrame
            The data used for training the estimator models and for estimating the
            causal effect.

        treatment : set or list of str, optional
            Set of names of treatments.

        outcome : set or list of str, optional
            Set of names of outcomes.

        identify_method : tuple of str or str, optional. Default to 'auto'.
            If the passed value is a tuple or list, then it should have two
            elements where the first one is 'backdoor' and the second is for
            the returned set style.

            Available options:
            'auto' : Find all possible backdoor adjustment set.
            ('backdoor', 'simple') or 'simple': Return the set of all direct confounders of
                                                both treatments and outcomes as a backdoor
                                                adjustment set.
            ('backdoor', 'minimal') or 'minimal': Return all possible backdoor adjustment sets with
                                                    minial number of elements.
            ('backdoor', 'all') or 'all': Return all possible backdoor adjustment sets.

        quantity : str, optional. Default to None
            The interested quantity when evaluating causal effects.

        Returns
        ----------
        ndarray or float
            The estimated causal effect in data.
        """
        # TODO: now only supports estimation with adjustment set. This needs
        # to be updated if the estimation of general identification problem
        # is solved.
        assert all((data is not None, treatment is not None, outcome is not None))

        # if estimator_model is None:
        #     estimator_model = DoubleML()
        assert estimator_model is not None

        # check columns
        check_cols(data, treatment, outcome)

        # check nodes in the graph
        ava_nodes = self.causal_graph.causation.keys()
        check_nodes(ava_nodes, treatment, outcome)

        if isinstance(identify_method, (list, tuple)):
            identify_method = identify_method[1]

        adj_set = self.get_backdoor_set(treatment, outcome, adjust=identify_method)

        self._est_result_dict = defaultdict(list)

        for sub_adj_set in adj_set:
            est_model = clone(estimator_model)

            est_model.fit(
                data=data,
                outcome=outcome,
                treatment=treatment,
                covariate=sub_adj_set,
                **kwargs,
            )

            self._est_result_dict["est_models"].append(est_model)

            effect = estimator_model.estimate(data=data, quantity=quantity)
            self._est_result_dict["effects"].append(effect)

        return self._est_result_dict

    def discover_graph(self, data):
        """Discover the causal graph from data.

        Parameters
        ----------
        data : pandas.DataFrame

        Returns
        ----------
        CausalGraph
        """
        assert data is not None, "Need data to call causal discovery."
        raise NotImplementedError

    def is_valid_backdoor_set(self, set_, treatment, outcome):
        """Determine if a given set is a valid backdoor adjustment set for
        causal effect of treatments on the outcomes.

        Parameters
        ----------
        set_ : set
            The adjustment set.

        treatment : set or list of str
            str is also acceptable for single treatment.

        outcome : set or list of str
            str is also acceptable for single outcome.

        Returns
        ----------
        bool
            True if the given set is a valid backdoor adjustment set for the
            causal effect of treatment on outcome in the current causal graph.
        """
        # TODO: improve the implementation
        # A valid backdoor set d-separates all backdoor paths between any pairs
        # of treatments and outcomes.
        treatment = set(treatment) if not isinstance(treatment, str) else {treatment}
        outcome = set(outcome) if not isinstance(outcome, str) else {outcome}

        # make sure all nodes are in the graph
        ava_nodes = self.causal_graph.causation.keys()
        check_nodes(ava_nodes, treatment, outcome, set_)

        # build the causal graph with unobserved variables explicit
        modified_dag = self.causal_graph.remove_outgoing_edges(
            treatment, new=True
        ).explicit_unob_var_dag

        return nx.d_separated(modified_dag, treatment, outcome, set_)

    def get_backdoor_set(self, treatment, outcome, adjust="simple", print_info=False):
        """Return the backdoor adjustment set for the given treatment and outcome.

        Parameters
        ----------
        treatment : set or list of str
            Names of the treatment. str is also acceptable for single treatment.

        outcome : set or list of str
            Names of the outcome. str is also acceptable for single outcome.

        adjust : str
            Set style of the backdoor set
                simple: directly return the parent set of treatment
                minimal: return the minimal backdoor adjustment set
                all: return all valid backdoor adjustment set.
        print_info : bool
            If True, print the identified results.

        Raises
        ----------
        Exception : IdentificationError
            Raise error if the style is not in simple, minimal or all or no
            set can satisfy the backdoor criterion.

        Returns
        ----------
        tuple
            The first element is the adjustment list, while the second is the
            encoded Prob.
        """
        # TODO: can I find the adjustment sets by using the adj matrix
        # TODO: improve the implementation

        # convert treatment and outcome to sets.
        treatment = set(treatment) if type(treatment) is not str else {treatment}
        outcome = set(outcome) if type(outcome) is not str else {outcome}

        # make sure all nodes are present
        ava_nodes = self.causal_graph.causation.keys()
        check_nodes(ava_nodes, outcome, treatment)

        modified_dag = self.causal_graph.remove_outgoing_edges(
            treatment, new=True
        ).explicit_unob_var_dag

        def determine(modified_dag, treatment, outcome):
            if all([self.causal_graph.causation[t] == [] for t in treatment]):
                return nx.d_separated(modified_dag, treatment, outcome, set())
            return True

        # raise exception if no set can be used as the backdoor set
        if not determine(modified_dag, treatment, outcome):
            print("No set can satisfy the backdoor criterion.")

            return None

        # Simply take all parents of both treatment and outcome as the backdoor set
        if adjust == "simple":
            # TODO: if the parent of x_i is the descendent of another x_j
            backdoor_list = []
            for t in treatment:
                backdoor_list += self.causal_graph.causation[t]
            adset = set(backdoor_list)
        # use other methods
        else:
            # Get all backdoor sets. currently implenmented
            # in a brutal force manner. NEED IMPROVEMENT
            des_set, backdoor_set_list = set(), []
            for t in treatment:
                des_set.update(nx.descendants(self.causal_graph.observed_dag, t))
            initial_set = set(list(ava_nodes)) - treatment - outcome - des_set

            if adjust == "minimal":
                # return the backdoor set which has the fewest number of elements
                for i in powerset(initial_set):
                    if nx.d_separated(modified_dag, treatment, outcome, i):
                        backdoor_set_list.append(i)
                        break
            else:
                backdoor_set_list = [
                    i
                    for i in powerset(initial_set)
                    if nx.d_separated(modified_dag, treatment, outcome, i)
                ]

            if not backdoor_set_list and not nx.d_separated(
                modified_dag, treatment, outcome, set()
            ):
                raise IdentificationError("No set can satisfy the backdoor criterion.")

            try:
                backdoor_set_list[0]
            except Exception:
                backdoor_set_list = [[]]
            finally:
                adset = set(backdoor_set_list[0])

            if any((adjust == "all", adjust == "minimal")):
                backdoor_list = backdoor_set_list
                self._adjustment_set = backdoor_set_list
            else:
                raise IdentificationError(
                    "Do not support backdoor set styles other than simple, all"
                    "or minimal."
                )

        # Build the corresponding probability distribution.
        product_expression = {
            Prob(variables=outcome, conditional=treatment.union(adset)),
            Prob(variables=adset),
        }
        prob = Prob(marginal=adset, product=product_expression)

        if print_info:
            print(f"The corresponding statistical estimand should be {prob.parse()})")
        return (backdoor_list, prob)

    def get_backdoor_path(self, treatment, outcome):
        """Return all backdoor paths connecting treatment and outcome.

        Parameters
        ----------
        treatment : str

        outcome : str

        Returns
        ----------
        list
            A list containing all valid backdoor paths between the treatment and
            outcome in the graph.
        """
        dag = self.causal_graph.explicit_unob_var_dag

        ava_nodes = self.causal_graph.causation.keys()
        check_nodes(ava_nodes, treatment, outcome)

        return [
            p
            for p in nx.all_simple_paths(dag.to_undirected(), treatment, outcome)
            if len(p) > 2 and p[1] in dag.predecessors(treatment)
        ]

    def has_collider(self, path, backdoor_path=True):
        """If the path in the current graph has a collider, return True, else
        return False.

        Parameters
        ----------
        path : list of str
            A list containing nodes in the path.

        backdoor_path : bool
            Whether the path is a backdoor path

        Returns
        ----------
        Boolean
            True if the path has a collider.
        """
        # TODO: improve the implementation.
        dag = self.causal_graph.explicit_unob_var_dag
        check_nodes(dag.nodes, path)

        if path[1] not in dag.predecessors(path[0]):
            backdoor_path = False

        if len(path) > 2:
            assert nx.is_path(dag.to_undirected(), path), "Not a valid path."

            j = 0
            if backdoor_path:
                if len(path) == 3:
                    return False

                for i, node in enumerate(path[1:], 1):
                    if node in dag.successors(path[i - 1]):
                        # if path[i] is the last element, then it is impossible
                        # to have collider
                        if i == len(path) - 1:
                            return False
                        j = i
                        break
                for i, node in enumerate(path[j + 1 :], j + 1):
                    if node in dag.predecessors(path[i - 1]):
                        return True
            else:
                for i, node in enumerate(path[1:], 1):
                    if node in dag.predecessors(path[i - 1]):
                        return True

        return False

    def is_connected_backdoor_path(self, path):
        """Test whether a backdoor path is connected.

        Parameters
        ----------
        path : list
            A list describing the path.

        Returns
        ----------
        Boolean
            True if path is a d-connected backdoor path and False otherwise.
        """
        check_node_graph = self.causal_graph.explicit_unob_var_dag
        check_nodes(check_node_graph.nodes, path)

        assert path[1] in check_node_graph.predecessors(path[0]), "Not a backdoor path."

        # A backdoor path is not connected if it contains a collider or not a
        # backdoor_path.
        # TODO: improve the implementation
        if self.has_collider(path) or path not in self.get_backdoor_path(
            path[0], path[-1]
        ):
            return False
        return True

    def is_frontdoor_set(self, set_, treatment, outcome):
        """Determine if the given set is a valid frontdoor adjustment set for the
        causal effect of treatment on outcome.

        Parameters
        ----------
        set_ : set of str

        treatement : str

        outcome : str

        Returns
        ----------
        Bool
            True if the given set is a valid frontdoor adjustment set for causal effects
            of treatments on outcomes.
        """
        ava_nodes = self.causal_graph.causation.keys()
        check_nodes(ava_nodes, set_, set(treatment), set(outcome))

        # rule 1, intercept all directed paths from treatment to outcome
        for path in nx.all_simple_paths(
            self.causal_graph.observed_dag, treatment, outcome
        ):
            if not any([path_node in set_ for path_node in path]):
                return False

        # rule 2, there is no unblocked back-door path from treatment to set_
        for subset in set_:
            for path in self.get_backdoor_path(treatment, subset):
                if self.is_connected_backdoor_path(path):
                    return False

        # rule 3, all backdoor paths from set_ to outcome are blocked by
        # treatment
        return self.is_valid_backdoor_set({treatment}, set_, {outcome})

    def get_frontdoor_set(self, treatment, outcome, adjust="simple"):
        """Return the frontdoor set for adjusting the causal effect between
        treatment and outcome.

        Parameters
        ----------
        treatment : set of str or str
            Name of the treatment. Should contain only one element.

        outcome : set of str or str
            Name of the outcome. Should contain only one element.

        Returns
        ----------
        tuple
            2 elements (adjustment_set, Prob)
        """
        ava_nodes = self.causal_graph.causation.keys()
        check_nodes(ava_nodes, set(treatment), set(outcome))

        if type(treatment) is set and type(outcome) is set:
            assert (
                len(treatment) == 1 and len(outcome) == 1
            ), "Treatment and outcome should be sets which contain one"
            "element for frontdoor adjustment."
            treatment, outcome = treatment.pop(), outcome.pop()

        # Find the initial set which will then be used to generate all
        # possible frontdoor sets with the method powerset()
        initial_set = set()
        for path in nx.all_simple_paths(
            self.causal_graph.observed_dag, treatment, outcome
        ):
            initial_set.update(set(path))
        initial_set = initial_set - {treatment} - {outcome}
        potential_set = powerset(initial_set)

        # Different frontdoor set styles.
        if adjust == "simple" or adjust == "minimal":
            adjustment, adset = set(), set()
            for set_ in potential_set:
                if self.is_frontdoor_set(set_, treatment, outcome):
                    adjustment.update(set_)
                    adset.update(set_)
                    break
        elif adjust == "all":
            adjustment = [
                i
                for i in powerset(initial_set)
                if self.is_frontdoor_set(i, treatment, outcome)
            ]
            adset = adjustment[0] if adjustment else set()
        else:
            raise IdentificationError(
                "The frontdoor set style must be one of simple, all" "and minimal."
            )

        if not adjustment:
            raise IdentificationError(
                "No set can satisfy the frontdoor adjustment criterion."
            )

        # Build the corresponding probability distribution.
        product_expression = {
            Prob(variables=adset, conditional={treatment}),
            Prob(
                marginal={treatment},
                product={
                    Prob(variables={outcome}, conditional={treatment}.union(adset)),
                    Prob(variables={treatment}),
                },
            ),
        }
        prob = Prob(marginal=adset, product=product_expression)
        return (adjustment, prob)

    def get_iv(self, treatment, outcome):
        """Find the instrumental variables for the causal effect of the
        treatment on the outcome.

        Parameters
        ----------
        treatment : iterable
            Name(s) of the treatment.

        outcome : iterable
            Name(s) of the outcome.

        Returns
        -------
        set
            A valid instrumental variable set which will be an empty one if
            there is no such set.
        """
        ava_nodes = self.causal_graph.causation.keys()
        check_nodes(ava_nodes, treatment, outcome)

        # 1. relevance: all possible instrument variables should be parents of the
        # treatment
        waited_instrument = set(self.causal_graph.parents(treatment))

        # 2. exclusion: build the graph where all incoming edges to the treatment
        # are removed such that those nodes which have effects on the outcome through
        # the treatment in the modified graph are excluded to be valid instruments
        exp_unob_graph = self.causal_graph.explicit_unob_var_dag
        modified_graph = remove_ingo_edges(exp_unob_graph, True, treatment)
        excluded_nodes_an = nx.ancestors(modified_graph, outcome)
        waited_instrument.difference_update(excluded_nodes_an)

        # We also should not count on descendents of ancestors of the outcome which
        # may have effects on y through backdoor paths
        # excluded_nodes_des = nx.descendants(modified_graph, excluded_nodes_an)
        excluded_nodes_des = descendents_of_iter(modified_graph, excluded_nodes_an)
        iv = waited_instrument.difference(excluded_nodes_des)

        if not iv:
            # print(f'No valid instrument variable has been found.')
            iv = None

        return iv

    def is_valid_iv(self, treatment, outcome, set_):
        """Determine whether a given set_ is a valid instrumental variable set.

        Parameters
        ----------
        treatment : iterable
            Name(s) of the treatment.

        outcome : iterable
            Name(s) of the outcome.

        set_ : iterable

        Returns
        -------
        bool
            True if the set is a valid instrumental variable set and False
            otherwise.
        """
        all_iv = self.get_iv(treatment, outcome)

        if not all_iv:
            raise Exception(
                "No valid instrument variable is found in the current causal model."
            )

        set_ = {set_} if isinstance(set_, str) else set_

        for element in set_:
            if not (element in all_iv):
                return False

        return True

    # The following method is for experimental use
    def estimate_hidden_cofounder(self, method="lr"):
        full_dag = self.causal_graph.explicit_unob_var_dag
        cg = deepcopy(self.causal_graph)
        all_nodes = full_dag.nodes
        hidden_cofounders = [node for node in all_nodes if "U" in node]
        estimated = 0

        def _compute_estimated(
            ob_ancestors,
            node,
            full_dag,
            hidden_ancestors,
        ):
            data_input = self.data[ob_ancestors]
            target = self.data[node]

            if method == "lr":
                reg = LR()
                reg.fit(data_input, target)
                predict = reg.predict(data_input)

            key_ = "estimate_" + hidden_ancestors[0].lower()
            nodes_ = nx.descendants(full_dag, hidden_ancestors[0])
            self.data[key_] = target - predict
            modified_nodes = list(nodes_)

            edge_list_del = []
            for i in range(len(modified_nodes)):
                for j in range(i + 1, len(modified_nodes)):
                    edge_list_del.append((modified_nodes[i], modified_nodes[j]))
                    edge_list_del.append((modified_nodes[j], modified_nodes[i]))

            cg.remove_edges_from(edge_list=edge_list_del, observed=False)
            cg.add_nodes([key_])
            edge_list_add = [(key_, i) for i in modified_nodes]
            cg.add_edges_from(edge_list=edge_list_add)
            estimated = estimated + 1

        if len(hidden_cofounders) == 0:
            print("Hidden cofounder estimation done. No hidden cofounder found.")
            return

        for node in all_nodes:
            if node not in hidden_cofounders:
                ancestors = nx.ancestors(full_dag, node)
                hidden_ancestors = [an for an in ancestors if an in hidden_cofounders]

                if len(hidden_ancestors) == 1:
                    ob_ancestors = [
                        an for an in ancestors if an not in hidden_cofounders
                    ]
                    _check_ls = [
                        check_ancestors_chain(full_dag, node, hidden_ancestors[0])
                        for node in ob_ancestors
                    ]
                    _true_num = sum(_check_ls)
                    if _true_num == len(ob_ancestors):
                        _compute_estimated(
                            ob_ancestors, node, full_dag, hidden_ancestors
                        )

        if estimated > 0:
            print(str(estimated) + " hidden cofounders estimated in this turn.")
            self.estimate_hidden_cofounder(method)
        else:
            print(
                "Hidden cofounder estimation done, and there could be "
                + str(len(hidden_cofounders))
                + " hidden cofounders remaining."
            )
            return

    def __repr__(self):
        return (
            f"A CausalModel for {self.causal_graph}, where the treatment is"
            f"{self.treatment} and the outcome is {self.outcome}."
        )
