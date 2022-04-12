"""
See (http://proceedings.mlr.press/v70/hartford17a/hartford17a.pdf) for
reference.

To use self-defined mixture density network and outcome network, one only
needs to define new MixtureDensityNetwork and OutcomeNet and wrap them with
MDNWrapper and OutcomeNetWrapper, respectively.
"""
from ast import arg
from operator import index
from ossaudiodev import control_labels
from re import L, S, X
import re
from tkinter import ON
from turtle import onclick
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from copy import deepcopy
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader
from torch.distributions import Categorical, Normal, MixtureSameFamily,\
    Independent

from .utils import GaussianProb, BatchData, convert2array, convert2tensor
from .base_models import BaseEstLearner, MLModel

# We first build the mixture density network.


class MixtureDensityNetwork(nn.Module):
    """
    See (https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf) for
    reference.
    """

    def __init__(self, in_d, out_d, hidden_d=512, num_gaussian=5):
        """
        Parameters
        ----------
        in_d : int
            Dimension of a single data point.
        out_d : int
            Dimension of the gaussian distribution.
        hidden_d : int
            Number of neurons in the hidden layer.
        num_gaussian : int
            Number of gaussian distributions to be mixed.
        """
        super().__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.hidden_d = hidden_d
        self.num_gaussian = num_gaussian
        self.hidden_layer = nn.Sequential(
            nn.Linear(in_d, hidden_d),
            nn.Tanh()
        )
        self.pi = nn.Linear(hidden_d, num_gaussian)
        self.sigma = nn.Linear(hidden_d, num_gaussian * out_d)
        self.mu = nn.Linear(hidden_d, num_gaussian * out_d)

    def foward(self, x):
        """
        Parameters
        ----------
        x : tensor
            Has shape of (b, in_d), where b is the batch size.

        Returns
        ----------
        pi : tensor
            Mixing coefficient with the shape (b, num_gaussian) and each
            component of pi is in the range [0, 1].
        mu : tensor
            Mean of each mixing component with the shape
            (b, num_gaussian, out_d).
        sigma : tensor
            Variance with the shape (b, num_gaussian, out_d) and each
            component of sigma is large than 0.
        """
        h = self.hidden_layer(x)
        pi = nn.Softmax(h)
        mu = self.mu(h).view(
            -1, self.num_gaussian, self.out_d
        )
        sigma = torch.exp(self.sigma(h)).view(
            -1, self.num_gaussian, self.out_d
        )
        return pi, mu, sigma

# To make the above MDN consistant to the standard machine learning models, we
# use the following wrapper to wrap it so that the methods such as fit() can
# be applied.


class MDNWrapper(MLModel):
    """
    Wrapped class for MixtureDensityNetwork.

    Attributes
    ----------
    mdn : MixtureDensityNetwork

    Methods
    ----------
    loss_fn(pi, mu, sigma, y)
        Calculate the loss used for training the mdn.
    fit(X, y, device='cuda', lr=0.01, epoch=1000,
        optimizer='SGD', batch_size=128)
        Train the mdn model with data (X, y).
    predict(X, y)
        Calculate the probability P(y|X) with the trained mixture density
        network.
    sample()
        Generate samples with the mixture density network.
    """

    def __init__(self, mdn):
        """
        Parameters
        ----------
        model : MixtureDensityNetwork
        """
        super().__init__()
        self.model = mdn
        self.in_d = mdn.in_d
        self.out_d = mdn.out_d
        self.num_gaussian = mdn.num_gaussian

    def loss_fn(self, pi, mu, sigma, y):
        """Calculate the loss used for training the mdn.

        Parameters
        ----------
        pi : tensor
            Has shape (b, num_gaussian) where b is the batch size and
            num_gaussian is the number of mixiing gaussian distributions. The
            mixing coefficient of gaussian distributions.
        mu : tensor
            Shape (b, num_gaussian, out_d) where out_d is the dimension of
            each mixed gaussian distribution.
        sigma : tensor
            Has shape (b, num_gaussian, out_d). The variance of the gaussian
            distributions.
        y : tensor
            Has shape (b, out_d).

        Returns
        ----------
        tensor
            The probability of taking value y in the probability distribution
            modeled by the mdn. Has the same shape as y.
        """
        gaussian_prob = GaussianProb(mu, sigma)
        p = gaussian_prob.mixture_density(pi, y)
        loss = torch.mean(
            -torch.log(p)
        )
        return loss

    def fit(self, X, y,
            device='cuda',
            lr=0.01,
            epoch=1000,
            optimizer='SGD',
            batch_size=128,
            **optim_config):
        """Train the mdn model with data (X, y).

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
        data = BatchData(X=X, y=y)
        train_loader = DataLoader(data, batch_size=batch_size)

        for e in range(epoch):
            for i, (X, y) in enumerate(train_loader):
                self.model.train()
                X, y = X.to(device), y.to(device)
                pi, mu, sigma = self.model(X)
                loss = self.loss_fn(pi, mu, sigma, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            print(f'End of epoch {e} | current loss {loss.data}')

    def predict(self, X, y):
        """Calculate the probability P(y|X) with the trained mixture density
            network.

        Parameters
        ----------
        X : tensor
            Has shape (b, in_d) where b is the batch size and in_d is the
            dimension of each data point
        y : tensor
            Has shape (b, out_d).

        Returns
        ----------
        tensor
            The probability density p(y|X) evaluated with the trained mdn.
        """
        pi, mu, sigma = self.model(X)
        gaussian_prob = GaussianProb(mu, sigma)
        p = gaussian_prob.mixture_density(pi, y)
        return p

    def _sample(self, X, sample_num):
        # TODO: remeber to call detach to depart from the calculatin graph
        """Generate a batch of sample according to the probability density returned by
            the MDN model.

        Parameters
        ----------
        X : tensor
            Shape (b, in_d) where b is the batch size.
        sample_num : tuple of int
            Eg., (5, ) means generating (5*b) samples.

        Returns
        ----------
        tensor
            Shape (b*sample_num, out_d).
        """
        pi, mu, sigma = self.model(X)
        mix = Categorical(pi)
        comp = Independent(Normal(mu, sigma))
        density = MixtureSameFamily(mix, comp)
        return density.sample(sample_num).view(-1, self.out_d)

    # def prob_one_hot(self, X):
    #     # TODO: possible implementation for one_hot vector.
    #     pi, mu, sigma = self.model(X)
    #     n = X.shape[0]

    #     # construct the conditional probability when y is a one-hot vector.
    #     id_matrix = torch.eye(self.out_d)
    #     pi_ = pi.repeat(self.out_d, 1)
    #     # x_ = X.repeat(self.out_d, 1)
    #     mu_ = mu.repeat(self.out_d, 1, 1)
    #     sigma_ = sigma.repeat(self.out_d, 1, 1)
    #     y = id_matrix[0].repeat(n, 1)
    #     # build a large y with size (n * out_d, out_d) where out_d is the
    #     # dimension of the one-hot vector y
    #     for i in range(1, self.out_d):
    #         y = torch.cat(
    #             (y, id_matrix[i].repeat(n, 1)), dim=0
    #         )
    #     gaussian_prob = GaussianProb(mu_, sigma_)
    #     # p = torch.sum(pi_ * gaussian_prob.prod_prob(y), dim=1)
    #     pass

# Then we build the outcome network.


class Net(nn.module):
    # TODO: add embedding layer
    def __init__(self, in_d, out_d, hidden_d1, hidden_d2=256, hidden_d3=256):
        super().__init__()
        self.fc1 = nn.Linear(in_d, hidden_d1)
        self.fc2 = nn.Linear(hidden_d1, hidden_d2)
        self.fc3 = nn.Linear(hidden_d2, hidden_d3)
        self.fc4 = nn.Linear(hidden_d3, out_d)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        output = self.fc4(x)
        return output

# Similarly, we wrap the above outcome network with a warpper.


class NetWrapper(MLModel):
    def __init__(self, y_net):
        super().__init__()
        self.model = y_net

    def fit(self, X, y, nn_torch=True, **kwargs):
        return super().fit(X, y, nn_torch, **kwargs)

    def predict(self, X):
        return super().predict(X)

    def _sample(self, sample_num):
        pass
# We are now ready to build the complete model with above wrapped outcome and
# treatment nets.


class DeepIV(BaseEstLearner):
    def __init__(
        self,
        x_net=None,
        y_net=None,
        z_d=None,
        x_d=None,
        y_d=None,
        w_d=None,
        x_hidden_d=None,
        y_hidden_d=None,
        num_gaussian=5,
        is_discrete_treatment=False,
        is_discrete_outcome=False,
        is_discrete_instrument=False,
        categories=None,
        random_state=2022,
    ):
        r"""Training of a DeepIV model g(x, w) is composed of 2 stages:
            1. In the first stage, we train a neural network to estimate the
            distribution of the treatment x given the instrument z and
            adjustment (probably also covariate) w;
            2. In the second stage, we train another neural network to estiamte
            the outcome y givn treatment x and adjustment (probably also
            covariate) w.
        The trained model is used to estimate the causal effect
            g(x_1, w) - g(x_0, w)
        or
            \partial_x g(x, w).

        Parameters
        ----------
        x_net : MDNWrapper or NetWrapper
            Representation of the mixture density network for continuous
            treatment or an usual classification net for discrete treatment.
        y_net :  NetWrapper
            Representation of the outcome network.
        """
        self.x_d = x_d
        self.w_d = w_d
        self.z_d = z_d
        self.y_d = y_d
        
        x_net_kwargs = {}
        if x_hidden_d is not None:
            x_net_kwargs['hidden_d'] = x_hidden_d
        if num_gaussian is not None:
            x_net_kwargs['num_gaussian'] = num_gaussian

        y_net_kwargs = {}
        if y_hidden_d is not None:
            y_net_kwargs['hidden_d'] = y_hidden_d

        self.x_net = self._gen_x_model(
            x_net,
            int(z_d+w_d),
            x_d,
            **x_net_kwargs
        )
        self.y_net = self._gen_y_model(
            y_net,
            int(x_d+w_d),
            y_d,
            **y_net_kwargs
        )

        super.__init__(
            random_state=random_state,
            is_discrete_treatment=is_discrete_treatment,
            is_discrete_outcome=is_discrete_outcome,
            is_discrete_instrument=is_discrete_instrument,
            categories=categories,
        )

    def fit(
        self,
        data,
        outcome,
        treatment,
        instrument,
        adjustment=None,
        approx_grad=True,
        sample_n=10,
        x_net_config=None,
        y_net_config=None,
        **kwargs
    ):
        """Train the DeepIV model.

        Parameters
        ----------
        z : tensor
            Instrument variables. Shape (b, z_d) where b is the batch size and
            z_d is the dimension of a single instrument variable data point.
        x : tensor
            Treatments. Shape (b, x_d).
        y : tensor
            Outcomes. Shape (b, y_d)
        w : tensor, defaults to None.
            Observed adjustments. Shape (b, w_d)
        sample_n : tuple of int
            Eg., (5, ) means generating (5*b) samples according to the
            probability density modeled by the x_net.
        discrete_treatment : bool
            If True, the x_net is chosen as the MixtureDensityNetwork.
        """
        self.outcome = outcome
        self.treatment = treatment
        self.adjustment = adjustment
        self.instrument = instrument

        if x_net_config is None and y_net_config is None:
            x_net_config = kwargs
            y_net_config = kwargs

        y, x, z, w = convert2array(
            data, outcome, treatment, instrument, adjustment
        )

        if self.is_discrete_treatment:
            if self.categories == 'auto' or self.categories is None:
                categories = 'auto'
            else:
                categories = list(self.categories)

            self.x_transformer = OneHotEncoder(categories=categories)
            self.x_transformer.fit(x)
            x = torch.tensor(
                self.x_transformer.transform(x).toarray()
            )

        if self.is_discrete_instrument:
            self._z_transformer = OneHotEncoder()
            self._z_transformer.fit(z)
            z = self._z_transformer.transform(z).toarray()

        if self.is_discrete_outcome:
            self.y_transformer = OneHotEncoder()
            self.y_transformer.fit(y)
            y = self.y_transformer.transform(y).toarray()

        y, x, w, z = convert2tensor(y, x, w, z)
        self.w = w
        # xnet_in has shape (n, z_d+w_d).
        x_net_in = torch.cat((z, w), dim=1) if w is not None else z

        # Step 1: train the model for estimating the treatment given the
        # instrument and adjustment
        self.x_net.fit(x_net_in, x, **x_net_config)

        # Step 2: generate new samples if calculating grad approximately
        if approx_grad:
            x_sampled = self.x_net._sample(x_net_in, sample_n)
        else:
            # TODO: the loss funcn should be modified if not approx_grad
            x_sampled = x

        if not self.is_discrete_treatment:
            w_sampled = w.repeat(sample_n[0], 1)
        else:
            w_sampled = w

        # build the samples for training of the second stage
        x_ = torch.cat((w_sampled, x_sampled), dim=1)

        # Step 3: fit the final counterfactual prediction model
        self.y_net.fit(x_, y, **y_net_config)

    def _prepare4est(
        self,
        data=None,
        treat=None,
        control=None,
        marginal_effect=False,
        *args,
        **kwargs
    ):
        treat = 1 if treat is None else treat
        control = 0 if control is None else control

        if data is None:
            w = self.w
        else:
            w = convert2tensor(
                convert2array(data, self.adjustment)[0]
            )[0]
        n = w.shape[0]
        ones = torch.eye(n, self.x_d)

        if self.discrete_treatment:
            # build the one_hot vector for treatment vector
            treat_id = (torch.ones(n, ) * treat).int()

            # build treatment vector xt and control vector x0
            xt = ones.index_select(dim=0, index=treat_id)
            x0 = ones.index_select(dim=0, index=torch.zeros(n, ).int())
            xt = torch.cat((w, xt), dim=1)
            x0 = torch.cat((w, x0), dim=1)

            return (self.y_net.predict(xt), self.y_net.predict(x0))
        else:
            xt = ones * treat
            x0 = ones * control
            xt = torch.cat((w, xt), dim=1)
            xt.requires_grad = True
            yt = self.y_net.predict(xt)

            if marginal_effect:
                return (xt.grad.detach(), )
            else:
                x0 = torch.cat((w, x0), dim=1)
                y0 = self.y_net.predict(x0)
                return (yt, y0)

    def estimate(
        self,
        data=None,
        treat=None,
        control=None,
        quantity='CATE',
        marginal_effect=False,
        *args,
        **kwargs,
    ):
        y_preds = self._prepare4est(
            data=data,
            treat=treat,
            control=control,
            marginal_effect=marginal_effect,
            *args,
            **kwargs
        )

        if not marginal_effect:
            yt, y0 = y_preds
        else:
            yt, y0 = y_preds[0], None

        if quantity == 'CATE' or quantity == 'ATE':
            return (yt - y0).mean(dim=0) if y0 is not None else yt.mean(dim=0)
        elif quantity == 'Counterfactual prediction':
            return yt

    def _gen_x_model(self, x_model, *args, **kwargs):
        if self.is_discrete_treatment:
            if x_model is None:
                assert any(args), 'Need parameters to define treatment net.'
                x_model = Net(*args, **kwargs)

            x_net = NetWrapper(x_model, **kwargs)
        else:
            if x_model is None:
                assert any(args), 'Need parameters to define treatment net.'
                x_model = MixtureDensityNetwork(*args)

            x_net = MDNWrapper(x_model)

        return x_net

    def _gen_y_model(self, y_model, *args):
        if y_model is None:
            assert any(args), 'Need parameters to define outcome net.'
            y_net = Net(*args)

        return NetWrapper(y_net)
