"""
See (http://proceedings.mlr.press/v70/hartford17a/hartford17a.pdf) for
reference.

To use self-defined mixture density network and outcome network, one only
needs to define new MixtureDensityNetwork and OutcomeNet and wrap them with
MDNWrapper and OutcomeNetWrapper, respectively.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from copy import deepcopy
from torch.utils.data import DataLoader
from torch.distributions import Categorical, Normal, MixtureSameFamily,\
    Independent

from .utils import GaussianProb, BatchData
from .base_models import BaseEstLearner, MLModel


class MixtureDensityNetwork(nn.Module):
    """
    See (https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf) for
    reference.
    """

    def __init__(self, in_d, out_d, hidden_d, num_gaussian):
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

    def sample(self, X, sample_num):
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


class OutcomeNet(nn.module):
    def __init__(self, in_d, out_d, hidden_d1, hidden_d2, hidden_d3):
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


class OutcomeNetWrapper(MLModel):
    def __init__(self, outcome_net):
        super().__init__()
        self.outcome_net = outcome_net


class DeepIV(BaseEstLearner):
    def __init__(self, treatment_net, outcome_net):
        """
        Parameters
        ----------
        treatment_net : MDNWrapper, optional
            Representation of the mixture density network.
        outcome_net : OutcomeNetWrapper
            Representation of the outcome network.
        """
        super().__init__()
        self.treatment_net = treatment_net
        self.outcome_net = outcome_net

    def fit(self, z, x, y, c=None, sample_n=None, discrete_treatment=False):
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
        c : tensor, defaults to None.
            Observed confounders. Shape (b, c_d)
        sample_n : tuple of int
            Eg., (5, ) means generating (5*b) samples according to the
            probability density modeled by the treatment_net.
        discrete_treatment : bool
            If True, the treatment_net is chosen as the MixtureDensityNetwork.
        """
        # TODO: can we use str type z, c, x, y to be consistent with other api?
        tnet_in = torch.cat((z, c), dim=1) if c is not None else z
        # tnet_in has shape (b, z_d+c_d).
        self.treatment_net.fit(tnet_in, x)
        x_ = self.treatment_net.sample(tnet_in, sample_n)
        c_ = c.repeat(sample_n[0], 1)
        x_ = torch.cat((c_, x_), dim=1)
        self.outcome_net.fit(x_, y, nn_torch=True, loss=nn.MSELoss())

    def prepare(self, data, outcome, treatment,
                confounder=None,
                individual=None,
                instrument=None,
                discrete_treatment=True):

        def convert_to_tensor(x):
            return torch.tensor(x.values)

        c = convert_to_tensor(data[confounder]) if confounder is not None \
            else None
        x = convert_to_tensor(data[treatment])
        y = convert_to_tensor(data[outcome])
        z = convert_to_tensor(data[instrument])
        self.fit(z, x, y, c)

        if individual:
            x = convert_to_tensor(individual[treatment])
            c = convert_to_tensor(individual[confounder])

        # TODO: binary treatment
        if discrete_treatment:
            x1, x0 = deepcopy(x), deepcopy(x)
            x1[:] = 1
            x0[:] = 0
            x1 = torch.cat((c, x1), dim=1)
            x0 = torch.cat((c, x0), dim=1)
            r = self.outcome_net.predict(x1) - self.outcome_net.predict(x0)
        else:
            x_ = torch.cat((c, x), dim=1)
            x_.requires_grad = True
            y = self.outcome_net.predict(x_)
            r = x_.grad.detach()[:, -self.treatment_net.out_d:]

        return r
