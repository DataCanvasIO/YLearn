import pandas as pd
import numpy as np

from copy import deepcopy

np.random.seed(2022)


class MetaLearner:
    def __init__(self, ml_model):
        self.ml_model = ml_model

    def prepare(self, data, outcome, treatment, adjustment, individual=None):
        """
        Prepare (fit the model) for estimating the quantities
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
        treatment : string
        adjustment : set or list
        condition_set : set or list
            The computation will be conducted in the group of data where
            variables in the condition_set have certain fixed values.
        """
        pass

    def estimate(self, data, outcome, treatment, adjustment, quantity='ATE',
                 condition_set=None, condition=None, individual=None):
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
        return self.prepare(
            data, outcome, treatment, adjustment=adjustment
        ).mean()

    def estimate_cate(self, data, outcome, treatment, adjustment,
                      condition_set, condition):
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
        self.ml_model = kwargs['ml_model']

    def prepare(self, data, outcome, treatment, adjustment, individual=None):
        """
        Prepare (fit the model) for estimating the quantities
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
        treatment : string
        adjustment : set or list
        """
        x = list(adjustment).append(treatment)
        self.model.fit(X=data[x], y=data[outcome])

        if individual:
            data = individual

        Xt1 = pd.DataFrame.copy(data)
        Xt1[treatment] = 1
        Xt0 = pd.DataFrame.copy(data)
        Xt0[treatment] = 0
        result = (
            self.ml_model.predict(Xt1) - self.ml_model.predict(Xt0)
        )
        return result


class TLearner(MetaLearner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ml_model_t1 = kwargs['ml_model']
        self.ml_model_t0 = deepcopy(kwargs['ml_model'])

    def prepare(self, data, outcome, treatment, adjustment, individual=None):
        """
        Prepare (fit the model) for estimating the quantities
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
        treatment : string
        adjustment : set or list
        """
        data_without_treatment = data.drop([treatment], axis=1)
        t1_data = data_without_treatment.loc[data[treatment] > 0]
        t0_data = data_without_treatment.loc[data[treatment] <= 0]
        self.ml_model_t1.fit(t1_data[adjustment], t1_data[outcome])
        self.ml_model_t0.fit(t0_data[adjustment], t0_data[outcome])

        if individual:
            data_without_treatment = individual.drop(list(treatment), axis=1)

        result = (
            self.ml_model_t1.predict(data_without_treatment) -
            self.ml_model_t0.predict(data_without_treatment)
        )
        return result


class XLearner(MetaLearner):
    def __init__(self) -> None:
        super().__init__()


class DragonNet(MetaLearner):
    def __init__(self) -> None:
        super().__init__()
