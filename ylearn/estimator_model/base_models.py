import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from ylearn.utils import to_repr
from .utils import BatchData
# from ..utils._common import check_cols

#TODO: add support for assigning different treatment values for different examples for all models.

class BaseEstModel:
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
    _prepare(data, outcome, treatment, adjustment, individual=None)
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

    def __init__(
        self,
        random_state=2022,
        is_discrete_treatment=False,
        is_discrete_outcome=False,
        _is_fitted=False,
        # is_discrete_instrument=False,
        categories='auto'
    ):
        self.random_state = random_state if random_state is not None else 2022
        self.is_discrete_treatment = is_discrete_treatment
        self.is_discrete_outcome = is_discrete_outcome
        self.categories = categories

        self._is_fitted = _is_fitted

        # fitted
        self.treatment = None
        self.outcome = None
        self.treats_ = None

    def fit(
        self,
        data,
        outcome,
        treatment,
        # adjustment=None,
        # covariate=None,
        **kwargs,
    ):
        assert data is not None and isinstance(data, pd.DataFrame)

        # check_cols(data, treatment, outcome)
        if isinstance(treatment, str):
            treatment = [treatment]

        self.treatment = treatment
        self.outcome = outcome

        for k, v in kwargs.items():
            setattr(self, k, v)
            # check_cols(data, v)

        if data is not None and self.is_discrete_treatment:
            treats = {}
            for t in treatment:
                treats[t] = tuple(np.sort(data[t].unique()).tolist())
        else:
            treats = None
        self.treats_ = treats

        return self

    def effect_nji(self, *args, **kwargs):
        raise NotImplementedError()

    #
    # def _prepare_(
    #     self,
    #     data,
    # ):
    #     pass
    #
    # def _prepare(
    #     self,
    #     data,
    #     outcome,
    #     treatment,
    #     adjustment,
    #     individual=None,
    #     **kwargs
    # ):
    #     # TODO: We should add a method like check_is_fitted.
    #     # This does not need the parameters like treatment. By calling this
    #     # function first, we then estimate quantites like ATE easily.
    #     r"""Prepare (fit the model) for estimating the quantities
    #         ATE: E[y|do(x_1)] - E[y|do(x_0)] = E_w[E[y|x=x_1,w] - E[y|x=x_0, w]
    #                                        := E_{adjustment}[
    #                                            Delta E[outcome|treatment,
    #                                                             adjustment]]
    #         CATE: E[y|do(x_1), z] - E[y|do(x_0), z] = E_w[E[y|x=x_1, w, z] -
    #                                                     E[y|x=x_0, w, z]]
    #         ITE: y_i(do(x_1)) - y_i(do(x_0))
    #         CITE: y_i(do(x_1))|z_i - y_i(do(x_0))|z_i
    #
    #     Parameters
    #     ----------
    #     data : DataFrame
    #     outcome : str
    #         Name of the outcome.
    #     treatment : str
    #         Name of the treatment.
    #     adjustment : set or list
    #         The adjutment set for the causal effect,
    #         i.e., P(outcome|do(treatment)) =
    #             \sum_{adjustment} P(outcome|treatment, adjustment)P(adjustment)
    #     individual : DataFrame, default to None
    #         The individual data for computing its causal effect.
    #
    #     Returns
    #     ----------
    #     np.array
    #     """
    #     pass

    def estimate(self, data=None, **kwargs):
        if not hasattr(self, '_is_fitted'):
            raise Exception('The estimator has not been fitted yet.')
    #
    # # def estimate(
    # #     self,
    # #     data,
    # #     outcome,
    # #     treatment,
    # #     adjustment,
    # #     quantity='ATE',
    # #     condition_set=None,
    # #     condition=None,
    # #     individual=None,
    # #     **kwargs
    # # ):
    # #     # This is the basic API for estimating a causal effect.
    #
    # #     # TODO: Note that we require that if parameters like treatment change
    # #     # then we should recall self.fit()
    # #     # method.
    # #     """General estimation method for quantities like ATE of
    # #     outcome|do(treatment).
    #
    # #     Parameter
    # #     ----------
    # #     data : DataFrame
    # #     outcome : str
    # #         Name of the outcome.
    # #     treatment : str
    # #         Name of the treatment.
    # #     adjustment : set or list
    # #         The valid adjustment set.
    # #     quantity: str
    # #         The type of desired quantity, including ATE, CATE, ITE and
    # #         CITE. Defaults to 'ATE'.
    # #     condition_set : set or list. Defaults to None
    # #     condition : list
    # #         A list whose length is the size of dataset and elements are
    # #         boolean such that we only perform the computation of
    # #         quantities if the corresponding element is True]. Defaults
    # #         to None.
    # #     individual : DataFrame. Defaults to None
    # #     kwargs : dict
    #
    # #     Raises
    # #     ----------
    # #     Exception
    # #         Raise exception if the quantity is not in ATE, CATE, ITE or CITE.
    #
    # #     Returns
    # #     ----------
    # #     float
    # #         The desired causal effect.
    # #     """
    # #     if quantity == 'ATE':
    # #         return self.estimate_ate(
    # #             data, outcome, treatment, adjustment, **kwargs
    # #         )
    # #     elif quantity == 'CATE':
    # #         return self.estimate_cate(
    # #             data, outcome, treatment, adjustment, condition_set,
    # #             condition, **kwargs
    # #         )
    # #     elif quantity == 'ITE':
    # #         return self.estimate_ite(
    # #             data, outcome, treatment, adjustment, individual, **kwargs
    # #         )
    # #     elif quantity == 'CITE':
    # #         return self.estimate_cite(
    # #             data, outcome, treatment, adjustment, condition_set,
    # #             condition, **kwargs
    # #         )
    # #     else:
    # #         raise Exception(
    # #             'Do not support estimation of quantities other'
    # #             'than ATE, CATE, ITE, or CITE'
    # #         )
    #
    # def estimate_ate(self, data, outcome, treatment, adjustment, **kwargs):
    #     """Estimate E[outcome|do(treatment=x1) - outcome|do(treatment=x0)]
    #
    #     Parameters
    #     ----------
    #     data : DataFrame
    #     outcome : str
    #         Name of the outcome.
    #     treatment : str
    #         Name of the treatment.
    #     adjustment : set or list of str
    #         The valid adjustment set.
    #
    #     Returns
    #     ----------
    #     float
    #     """
    #     return self._prepare(
    #         data, outcome, treatment, adjustment=adjustment, **kwargs
    #     ).mean()
    #
    # def estimate_cate(self, data, outcome, treatment, adjustment,
    #                   condition_set, condition, **kwargs):
    #     """Estimate E[outcome|do(treatment=x1), condition_set=condition
    #                 - outcome|do(treatment=x0), condition_set=condition]
    #
    #     Parameters
    #     ----------
    #     data : DataFrame
    #     outcome : str
    #         Name of the outcome.
    #     treatment : str
    #         Name of the treatment.
    #     adjustment : set or list of str
    #         The valid adjustment set.
    #     condition_set : set of str
    #     condition : boolean
    #         The computation will be performed only using data where conition is
    #         True.
    #
    #     Returns
    #     ----------
    #     float
    #     """
    #     # TODO: considering let the final model also be the function of
    #     # conditional variables.
    #     # modify the data for estmating cate.
    #     assert condition_set is not None, \
    #         'Need an explicit condition set to perform the analysis.'
    #
    #     new_data = data.loc[condition].drop(list(condition_set), axis=1)
    #     cate = self.estimate_ate(
    #         new_data, outcome, treatment, adjustment, **kwargs
    #     )
    #     return cate
    #
    # def estimate_ite(self, data, outcome, treatment, adjustment,
    #                  individual, **kwargs):
    #     assert individual is not None, \
    #         'Need an explicit individual to perform computation of individual'
    #     'causal effect.'
    #
    #     return self._prepare(
    #         data, outcome, treatment, adjustment,
    #         individual=individual, **kwargs
    #     )
    #
    # def estimate_cite(self, data, outcome, treatment, adjustment,
    #                   condition_set, condition, individual, **kwargs):
    #     assert individual is not None, \
    #         'Need an explicit individual to perform computation of individual'
    #     'causal effect.'
    #     assert condition_set is not None, \
    #         'Need an explicit condition set to perform the analysis.'
    #
    #     new_data = data.loc[condition].drop(list(condition_set), axis=1)
    #     cite = self._prepare(
    #         new_data, outcome, treatment, adjustment, individual, **kwargs
    #     )
    #     return cite
    #
    # def counterfactual_prediction(self):
    #     pass

    def __repr__(self):
        return to_repr(self)


class MLModel:
    """
    A parent class for possible new machine learning models which are not
    supported by sklearn.
    """

    def __init__(self, model):
        """
        Parameters
        ----------
        model : nn, optional
            This can be any machine learning models.
        """
        self.model = model

    def fit(self, X, y, nn_torch=True, **kwargs):
        if nn_torch:
            self._fit_nn_torch(X, y, **kwargs)
        else:
            # define this for other types of fit functions
            pass

    def _fit_nn_torch(
        self,
        X, y,
        device='cuda',
        lr=0.01,
        epoch=1000,
        optimizer='SGD',
        batch_size=64,
        **optim_config
    ):
        """Train the nn model with data (X, y).

        Parameters
        ----------
        X : tensor
            Has shape (b, in_d) where b is the batch size or the number of data
            points and in_d is the dimension of each data point.
        y : tensor
            Has shape (b, out_d) where out_d is the dimension of each y.
        device : str, optional. Defaults to 'cuda'.
        lr : float, optional. Defaults to 0.01
            Learning rate.
        epoch : int, optional. Defaults to 1000
            The number of epochs used for training.
        optimizer : str, optional. Defaults to 'SGD'
            Currently including SGD and Adam The type of optimizer used for
            training.
        batch_size: int, optional. Defaults to 128
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

    def predict_proba(self, X, target):
        pass

    def fit_predict(self, X, y, X_test=None, **kwargs):
        self.fit(X, y, **kwargs)

        if X_test is None:
            X_test = X

        return self.predict(X_test)


# Modify this if you want to use networks with other structures.
class MultiClassNet(nn.Module):
    def __init__(self,
                 in_d,
                 out_d,
                 hidden_d1=100,
                 hidden_d2=128,
                 hidden_d3=64):
        super().__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.fc1 = nn.Linear(in_d, hidden_d1)
        self.fc2 = nn.Linear(hidden_d1, hidden_d2)
        self.fc3 = nn.Linear(hidden_d2, hidden_d3)
        self.fc4 = nn.Linear(hidden_d3, out_d)

    def fowrad(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.relu(x)
        return output


class MCNWrapper(MLModel):
    def __init__(self, treatment_net):
        super().__init__()
        self.model = treatment_net

    def fit(self, X, y, **optim_config):
        loss = nn.CrossEntropyLoss()
        self._fit_nn_torch(X, y, loss=loss, **optim_config)

    def predict(self, X, label=True):
        y_pred = nn.Softmax(dim=1)(self.model(X))
        if label:
            y_pred = y_pred.argmax(dim=1).view(X.shape[0], -1)
        return y_pred

    def predict_proba(self, X, target=None):
        y_pred = self.predict(X, label=False)
        if target is None:
            return y_pred
        else:
            return y_pred[:, target]

    def sample(self, X, sample_num):
        # This method is unnecessary for many tasks.
        pass
