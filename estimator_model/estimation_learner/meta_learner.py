from _typeshed import SupportsDunderLE
import pandas as pd
import numpy as np

from sklearn import linear_model
from copy import deepcopy

np.random.seed(2022)


class MetaLearner:
    """
    Base class for various metalearner.

    Attributes
    ----------

    Methods
    ----------
    prepare(data, outcome, treatment, adjustment, individual=None)
        Prepare (fit the model) for estimating various quantities including
        ATE, CATE, ITE, and CITE.
    estimate(data, outcome, treatment, adjustment, quantity='ATE',
                 condition_set=None, condition=None, individual=None)
        Integrate estimations for various quantities into a single method.
    estimate_ate(self, data, outcome, treatment, adjustment)
    estimate_cate(self, data, outcome, treatment, adjustment,
                      condition_set, condition)
    estimate_ite(self, data, outcome, treatment, adjustment, individual)
    estimate_cite(self, data, outcome, treatment, adjustment,
                      condition_set, condition, individual)
    """

    def __init__(self):
        self.ml_model_dic = {
            'LR': linear_model.LinearRegression()
        }

    def prepare(self, data, outcome, treatment, adjustment, individual=None):
        """Prepare (fit the model) for estimating the quantities
            ATE: E[y|do(x_1)] - E[y|do(x_0)] = E_w[E[y|x=x_1,w] - E[y|x=x_0, w]
                                           := E_{adjustment}[
                                               Delta E[outcome|treatment,
                                                                adjustment]]
            CATE: E[y|do(x_1), z] - E[y|do(x_0), z] = E_w[E[y|x=x_1, w, z] -
                                                        E[y|x=x_0, w, z]]
            ITE: y_i(do(x_1)) - y_i(do(x_0))
            CITE: y_i(do(x_1))|z_i - y_i(do(x_0))|z_i

        Parameters
        ----------
        data : DataFrame
        outcome : str
            Name of the outcome.
        treatment : str
            Name of the treatment.
        adjustment : set or list
            The adjutment set for the causal effect,
            i.e., P(outcome|do(treatment)) =
                \sum_{adjustment} P(outcome|treatment, adjustment)P(adjustment)
        individual : DataFrame, default to None
            The individual data for computing its causal effect.

        Returns
        ----------
        np.array
        """
        pass

    def estimate(self, data, outcome, treatment, adjustment, quantity='ATE',
                 condition_set=None, condition=None, individual=None):
        """General estimation method for quantities like ATE of
        outcome|do(treatment).

        Parameter
        ----------
        data : DataFrame
        outcome : str
            Name of the outcome.
        treatment : str
            Name of the treatment.
        adjustment : set or list
            The valid adjustment set.
        quantity: str
            The type of desired quantity, including ATE, CATE, ITE and
            CITE. Defaults to 'ATE'.
        condition_set : set or list. Defaults to None.
        condition : list
            A list whose length is the size of dataset and elements are
            boolean such that we only perform the computation of
            quantities if the corresponding element is True]. Defaults
            to None.
        individual : DataFrame. Defaults to None.

        Raises
        ----------
        Exception
            Raise exception if the quantity is not in ATE, CATE, ITE or CITE.

        Returns
        ----------
        float
            The desired causal effect.
        """
        if quantity == 'ATE':
            return self.estimate_ate(data, outcome, treatment, adjustment)
        elif quantity == 'CATE':
            return self.estimate_cate(
                data, outcome, treatment, adjustment, condition_set, condition
            )
        elif quantity == 'ITE':
            return self.estimate_ite(data, outcome, treatment, adjustment)
        elif quantity == 'CITE':
            return self.estimate_cite(
                data, outcome, treatment, adjustment, condition_set, condition
            )
        else:
            raise Exception(
                'Do not support estimation of quantities other'
                'than ATE, CATE, ITE, or CITE'
            )

    def estimate_ate(self, data, outcome, treatment, adjustment):
        """Estimate E[outcome|do(treatment=t1) - outcome|do(treatment=t0)]

        Parameters
        ----------
        data : DataFrame
        outcome : str
            Name of the outcome.
        treatment : str
            Name of the treatment.
        adjustment : set or list
            The valid adjustment set.

        Returns
        ----------
        float
        """
        return self.prepare(
            data, outcome, treatment, adjustment=adjustment
        ).mean()

    def estimate_cate(self, data, outcome, treatment, adjustment,
                      condition_set, condition):
        """Estimate E[outcome|do(treatment=t1), condition_set=condition
                    - outcome|do(treatment=t0), condition_set=condition]

        Parameters
        ----------
        data : DataFrame
        outcome : str
            Name of the outcome.
        treatment : str
            Name of the treatment.
        adjustment : set or list
            The valid adjustment set.
        condition_set : set
        condition : boolean
            The computation will be performed only using data where conition is
            True.

        Returns
        ----------
        float
        """
        # modify the data for estmating cate.
        new_data = data.loc[condition].drop(list(condition_set), axis=1)
        cate = self.prepare(new_data, outcome, treatment, adjustment).mean()
        return cate

    def estimate_ite(self, data, outcome, treatment, adjustment, individual):
        assert individual is not None, 'Need an explicit individual to perform'
        'computation of individual causal effect.'

        return self.prepare(
            data, outcome, treatment, adjustment, individual=individual
        )

    def estimate_cite(self, data, outcome, treatment, adjustment,
                      condition_set, condition, individual):
        assert individual is not None, 'Need an explicit individual to perform'
        'computation of individual causal effect.'

        new_data = data.loc[condition].drop(list(condition_set), axis=1)
        cite = self.prepare(
            new_data, outcome, treatment, adjustment, individual
        )
        return cite

    def counterfactual_prediction(self):
        pass


class SLearner(MetaLearner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        try:
            self.ml_model = self.ml_model_dic[kwargs['ml_model']]
        except Exception:
            self.ml_model = kwargs['ml_model']

    def prepare(self, data, outcome, treatment, adjustment, individual=None):
        """Prepare (fit the model) for estimating the quantities
            ATE: E[y|do(x_1)] - E[y|do(x_0)] = E_w[E[y|x=x_1,w] - E[y|x=x_0, w]
                                           := E_{adjustment}[
                                               Delta E[outcome|treatment,
                                                                adjustment]]
            CATE: E[y|do(x_1), z] - E[y|do(x_0), z] = E_w[E[y|x=x_1, w, z] -
                                                        E[y|x=x_0, w, z]]
            ITE: y_i(do(x_1)) - y_i(do(x_0))
            CITE: y_i(do(x_1))|z_i - y_i(do(x_0))|z_i

        Parameters
        ----------
        data : DataFrame
        outcome : string
            Name of the outcome.
        treatment : string
            Name of the treatment.
        adjustment : set or list
            The adjutment set for the causal effect,
            i.e., P(outcome|do(treatment)) =
                \sum_{adjustment} P(outcome|treatment, adjustment)P(adjustment)
        individual : DataFrame, default to None
            The individual data for computing its causal effect.

        Returns
        ----------
        np.array
        """
        x = list(adjustment)
        x.append(treatment)
        self.ml_model.fit(X=data[x], y=data[outcome])

        if individual:
            data = individual

        t1_data = pd.DataFrame.copy(data)
        t1_data[treatment] = 1
        t0_data = pd.DataFrame.copy(data)
        t0_data[treatment] = 0
        result = (
            self.ml_model.predict(t1_data[x]) -
            self.ml_model.predict(t0_data[x])
        )
        return result


class TLearner(MetaLearner):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model = kwargs['ml_model']

        if type(model) is str:
            model = self.ml_model_dic[model]

        self.ml_model_t1 = model
        self.ml_model_t0 = deepcopy(model)

    def prepare(self, data, outcome, treatment, adjustment, individual=None):
        """Prepare (fit the model) for estimating the quantities
            ATE: E[y|do(x_1)] - E[y|do(x_0)] = E_w[E[y|x=x_1,w] - E[y|x=x_0, w]
                                           := E_{adjustment}[
                                               Delta E[outcome|treatment,
                                                                adjustment]]
            CATE: E[y|do(x_1), z] - E[y|do(x_0), z] = E_w[E[y|x=x_1, w, z] -
                                                        E[y|x=x_0, w, z]]
            ITE: y_i(do(x_1)) - y_i(do(x_0))
            CITE: y_i(do(x_1))|z_i - y_i(do(x_0))|z_i

        Parameters
        ----------
        data : DataFrame
        outcome : string
            Name of the outcome.
        treatment : string
            Name of the treatment.
        adjustment : set or list
            The adjutment set for the causal effect,
            i.e., P(outcome|do(treatment)) =
                \sum_{adjustment} P(outcome|treatment, adjustment)P(adjustment)
        individual : DataFrame, default to None
            The individual data for computing its causal effect.

        Returns
        ----------
        np.array
        """
        data_without_treatment = data.drop([treatment], axis=1)
        t1_data = data_without_treatment.loc[data[treatment] > 0]
        t0_data = data_without_treatment.loc[data[treatment] <= 0]
        self.ml_model_t1.fit(t1_data[adjustment], t1_data[outcome])
        self.ml_model_t0.fit(t0_data[adjustment], t0_data[outcome])

        if individual:
            data_ = individual[adjustment]
        else:
            data_ = data[adjustment]

        result = (
            self.ml_model_t1.predict(data_) - self.ml_model_t0.predict(data_)
        )
        return result


class XLearner(MetaLearner):
    """
    The XLearner is composed of 3 steps:
        1. Train two different models for the control group and treated group
            f_0(w), f_1(w)
        2. Generate two new datasets (h_0, w) using the control group and
            (h_1, w) using the treated group where
            h_0 = f_1(w) - y_0(w), h_1 = y_1(w) - f_0(w). Then train two models
            k_0(w) and k_1(w) in these datasets.
        3. Get the final model using the above two models:
            g(w) = k_0(w)a(w) + k_1(w)(1 - a(w)).
    Finally,  we estimate the ATE as follows:
        ATE = E_w(g(w)).
    See Kunzel, et al., (https://arxiv.org/abs/1706.03461) for reference.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model = kwargs['ml_model']

        if type(model) is str:
            model = self.ml_model_dic[model]

        self.f1 = model
        self.f0 = deepcopy(model)
        self.k1 = deepcopy(model)
        self.k0 = deepcopy(model)

    def prepare(self, data, outcome, treatment, adjustment, individual=None):
        """Prepare (fit the model) for estimating the quantities
            ATE: E[y|do(x_1)] - E[y|do(x_0)] = E_w[E[y|x=x_1,w] - E[y|x=x_0, w]
                                           := E_{adjustment}[
                                               Delta E[outcome|treatment,
                                                                adjustment]]
            CATE: E[y|do(x_1), z] - E[y|do(x_0), z] = E_w[E[y|x=x_1, w, z] -
                                                        E[y|x=x_0, w, z]]
            ITE: y_i(do(x_1)) - y_i(do(x_0))
            CITE: y_i(do(x_1))|z_i - y_i(do(x_0))|z_i

        Parameters
        ----------
        data : DataFrame
        outcome : string
            Name of the outcome.
        treatment : string
            Name of the treatment.
        adjustment : set or list
            The adjutment set for the causal effect,
            i.e., P(outcome|do(treatment)) =
                \sum_{adjustment} P(outcome|treatment, adjustment)P(adjustment)
        individual : DataFrame, default to None
            The individual data for computing its causal effect.
        """
        # step 1
        data_without_treatment = data.drop([treatment], axis=1)
        t1_data = data_without_treatment.loc[data[treatment] > 0]
        t0_data = data_without_treatment.loc[data[treatment] <= 0]
        self.f1.fit(t1_data[adjustment], t1_data[outcome])
        self.f0.fit(t0_data[adjustment], t0_data[outcome])

        # setp 2
        h1_data = t1_data.drop(outcome, axis=1)
        h0_data = t0_data.drop(outcome, axis=1)
        h1 = t1_data[outcome] - self.f0.predict(h1_data[adjustment])
        h0 = self.f1.predict(h0_data[adjustment]) - t0_data[outcome]
        self.k1.fit(h1_data[adjustment], h1)
        self.k0.fit(h0_data[adjustment], h0)

        # step 3
        if individual:
            data_ = individual[adjustment]
        else:
            data_ = data[adjustment]
        # TODO: more choices of rho
        rho = 0.5
        result = rho * self.k1.predict(data_) - \
            (1 - rho) * self.k0.predict(data_)
        return result


# class DragonNet(MetaLearner):
#     """
#     See Shi., et al., (https://arxiv.org/pdf/1906.02120.pdf) for reference.

#     Args:
#         MetaLearner ([type]): [description]
#     """

#     def __init__(self) -> None:
#         super().__init__()
