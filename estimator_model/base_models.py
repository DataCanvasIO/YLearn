import torch
import torch.nn as nn
import torch.optim as optim

from copy import deepcopy
from sklearn import linear_model
from torch.utils.data import DataLoader

from .utils import BatchData


class BaseEstLearner:
    """
    Base class for various estimation learner.

    Attributes
    ----------
    ml_model_dic : dict
        A dictionary of default machine learning sklearn models currently
        including
            'LR': LinearRegression
            'LogistR': LogisticRegression.

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

    def prepare(self, data, outcome, treatment, adjustment,
                individual=None, **kwargs):
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

    def estimate(self, data, outcome, treatment, adjustment,
                 quantity='ATE',
                 condition_set=None,
                 condition=None,
                 individual=None,
                 **kwargs):
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
            return self.estimate_ate(
                data, outcome, treatment, adjustment, **kwargs
            )
        elif quantity == 'CATE':
            return self.estimate_cate(
                data, outcome, treatment, adjustment, condition_set,
                condition, **kwargs
            )
        elif quantity == 'ITE':
            return self.estimate_ite(
                data, outcome, treatment, adjustment, individual, **kwargs
            )
        elif quantity == 'CITE':
            return self.estimate_cite(
                data, outcome, treatment, adjustment, condition_set,
                condition, **kwargs
            )
        else:
            raise Exception(
                'Do not support estimation of quantities other'
                'than ATE, CATE, ITE, or CITE'
            )

    def estimate_ate(self, data, outcome, treatment, adjustment, **kwargs):
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
            data, outcome, treatment, adjustment=adjustment, **kwargs
        ).mean()

    def estimate_cate(self, data, outcome, treatment, adjustment,
                      condition_set, condition, **kwargs):
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
        cate = self.estimate_ate(
            new_data, outcome, treatment, adjustment, **kwargs)
        return cate

    def estimate_ite(self, data, outcome, treatment, adjustment,
                     individual, **kwargs):
        assert individual is not None, \
            'Need an explicit individual to perform computation of individual'
        'causal effect.'

        return self.prepare(
            data, outcome, treatment, adjustment,
            individual=individual, **kwargs
        )

    def estimate_cite(self, data, outcome, treatment, adjustment,
                      condition_set, condition, individual, **kwargs):
        assert individual is not None, \
            'Need an explicit individual to perform computation of individual'
        'causal effect.'

        assert condition_set is not None, \
            'Need an explicit condition set to perform the analysis.'

        new_data = data.loc[condition].drop(list(condition_set), axis=1)
        cite = self.prepare(
            new_data, outcome, treatment, adjustment, individual, **kwargs
        )
        return cite

    def counterfactual_prediction(self):
        pass


class MLModel:
    """
    A parent class for possible new machine learning models which are not
    supported by sklearn.
    """

    def __init__(self, model):
        self.model = model

    def fit(self, X, y, nn_torch=False, **kwargs):
        if nn_torch:
            self._fit_nn_torch(X, y, **kwargs)
        else:
            pass

    def _fit_nn_torch(self, X, y,
                      device='cuda',
                      lr=0.01,
                      epoch=1000,
                      optimizer='SGD',
                      batch_size=128,
                      **optim_config):
        """Train the nn model with data (X, y).

        Parameters
        ----------
        X : tensor
            Has shape (b, in_d) where b is the batch size or the number of data
            points and in_d is the dimension of each data point.
        y : tensor
            Has shape (b, out_d) where out_d is the dimension of each y.
        device : str, optional. Defaults to 'cuda'.
        lr : float, optional. Defaults to 0.01.
            Learning rate.
        epoch : int, optional. Defaults to 1000.
            The number of epochs used for training.
        optimizer : str, optional. Defaults to 'SGD'.
            Currently including SGD and Adam The type of optimizer used for
            training.
        batch_size: int, optional. Defaults to 128.
        optim_config : other parameters for various optimizers.
        """
        self.model = self.model.to(device)
        op = {
            'SGD': optim.SGD(self.model.parameters(), lr=lr),
            'Adam': optim.Adam(self.model.parameters(), lr=lr, **optim_config)
        }
        opt = op[optimizer]
        loss_fn = optim_config['loss']
        data = BatchData(X=X, y=y)
        train_loader = DataLoader(data, batch_size=batch_size)

        for e in range(epoch):
            for i, (X, y) in enumerate(train_loader):
                self.model.train()
                X, y = X.to(device), y.to(device)
                y_predict = self.model(X)
                loss = loss_fn(y_predict, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            print(f'End of epoch {e} | current loss {loss.data}')

    def predict(self, X):
        return self.model(X)

    def fit_predict(self, X, y, **kwargs):
        self.fit(X, y, **kwargs)
        return self.predict(X)


# class EgModel(MLModel):
#     """
#     An example class for constructing a new machine learning model to be used
#     in YLearn.
#     """

#     def __init__(self) -> None:
#         super().__init__()

#     def fit(self, X, y):
#         return super().fit(X, y)
