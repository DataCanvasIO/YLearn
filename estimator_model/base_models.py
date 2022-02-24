from sklearn import linear_model


class BaseEstLearner:
    """
    Base class for various estimation learner.

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
            'LR': linear_model.LinearRegression(),
            'LogistR': linear_model.LogisticRegression(),
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
            assert individual is not None, \
                'Need an explicit individual to perform the analysis.'
            return self.estimate_ite(
                data, outcome, treatment, adjustment, individual
            )
        elif quantity == 'CITE':
            assert condition_set is not None, \
                ''
            return self.estimate_cite(
                data, outcome, treatment, adjustment, condition_set, condition
            )
        else:
            raise Exception(
                'Do not support estimation of quantities other'
                'than ATE, CATE, ITE, or CITE'
            )

    def estimate_ate(self, data, outcome, treatment, adjustment):
        """Estimate E[outcome|do(treatment=x1) - outcome|do(treatment=x0)]

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
        """Estimate E[outcome|do(treatment=x1), condition_set=condition
                    - outcome|do(treatment=x0), condition_set=condition]

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
        assert condition_set is not None, \
            'Need an explicit condition set to perform the analysis.'

        new_data = data.loc[condition].drop(list(condition_set), axis=1)
        cate = self.estimate_ate(new_data, outcome, treatment, adjustment)
        return cate

    def estimate_ite(self, data, outcome, treatment, adjustment, individual):
        assert individual is not None, \
            'Need an explicit individual to perform computation of individual'
        'causal effect.'

        return self.prepare(
            data, outcome, treatment, adjustment, individual=individual
        )

    def estimate_cite(self, data, outcome, treatment, adjustment,
                      condition_set, condition, individual):
        assert individual is not None, \
            'Need an explicit individual to perform computation of individual'
        'causal effect.'

        assert condition_set is not None, \
            'Need an explicit condition set to perform the analysis.'

        new_data = data.loc[condition].drop(list(condition_set), axis=1)
        cite = self.prepare(
            new_data, outcome, treatment, adjustment, individual
        )
        return cite

    def counterfactual_prediction(self):
        pass


class MlModels:
    def __init__(self) -> None:
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass


class NewEgModel(MlModels):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y):
        return super().fit(X, y)
