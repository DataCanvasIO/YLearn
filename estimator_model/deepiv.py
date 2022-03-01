"""
See (http://proceedings.mlr.press/v70/hartford17a/hartford17a.pdf) for
reference.
"""
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

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


class MDNWrapped(MLModel):
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
        mdn : MixtureDensityNetwork
        """
        super().__init__()
        self.mdn = mdn

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
        p = pi * gaussian_prob.prod_prob(y)
        loss = torch.mean(
            -torch.log(torch.sum(p, dim=1))
        )
        return loss

    def fit(self, X, y,
            device='cuda',
            lr=0.01,
            epoch=1000,
            optimizer='SGD',
            batch_size=128):
        """Train the mdn model with data (X, y).

        Parameters
        ----------
        X : tensor
            Has shape (b, in_d) where b is the batch size and in_d is the
            dimension of each data point.
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
        """
        self.mdn = self.mdn.to(device)
        op = {
            'SGD': optim.SGD(self.mdn.parameters(), lr=lr),
            'Adam': optim.Adam(self.mdn.parameters(), lr=lr)
        }
        opt = op[optimizer]
        data = BatchData(X=X, y=y)
        train_loader = DataLoader(data, batch_size=batch_size)

        for e in range(epoch):
            for i, (X, y) in enumerate(train_loader):
                self.mdn.train()
                X, y = X.to(device), y.to(device)
                pi, mu, sigma = self.mdn(X)
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
            The probability P(y|X) evaluated with the trained mdn.
        """
        pi, mu, sigma = self.mdn(X)
        gaussian_prob = GaussianProb(mu, sigma)
        p = pi * gaussian_prob.prod_prob(y)
        return p

    def sample(self):
        pass
