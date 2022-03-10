import torch
import math

import torch.nn as nn
import numpy as np

from torch.distributions import Categorical, Independent, MixtureSameFamily, \
    Normal
from torch.utils.data import Dataset


class BatchData(Dataset):
    def __init__(self, X=None, y=None, X_test=None, y_test=None, train=True):
        if train:
            self.data = X
            self.target = y
        else:
            self.data = X_test
            self.target = y_test

    def __len__(self):
        return self.target.shape[1]

    def __getitem__(self, index):
        return self.data[:, index], self.target[:, index]


class GaussianProb:
    """
    A class for gaussian distribution.

    Attributes
    ----------
    mu : tensor
    sigma : tensor

    Methods
    ----------
    prob(x)
        Return the probability of taking x.
    prod_prob(x)
        Return prob where elements in the last dimension are producted.
    """

    def __init__(self, mu, sigma):
        """
        Parameters
        ----------
        mu : tensor
            Mean of the gaussian distribution with shape
            (b, num_gaussian, out_d), where b is the batch size, num_guassian
            is the number of the mixing components, and out_d is the dimension
            of per gaussian distribution.
        sigma : tensor
            Variance of the gaussian distribution with shape
            (b, num_gaussian, out_d), where b is the batch size, num_guassian
            is the number of the mixing components, and out_d is the dimension
            of per gaussian distribution.
        """
        self.mu = mu
        self.sigma = sigma

    def prob_density(self, y):
        """Return the probability of taking y.

        Parameters
        ----------
        y : tensor
            Shape (b, out_d) where b is the batch size.
        Returns
        ----------
        tensor
            The shape is the same as that of mu.
        """
        y = y.unsqueeze(dim=1).expand_as(self.mu)
        p = (1.0 / math.sqrt(2 * math.pi)) * torch.exp(
            -0.5 * ((y - self.mu) / self.sigma)**2
        ) / self.sigma
        return p

    def mixture_density(self, pi, y, torch_D=False):
        """Can replace the implementation with the one provided by pytorch, see
        (https://pytorch.org/docs/stable/distributions.html#mixturesamefamily)
            for details.
        """
        if torch_D:
            mix = Categorical(pi)
            comp = Independent(Normal(self.mu, self.sigma))
            density = MixtureSameFamily(mix, comp).log_prob(y)
        else:
            p_k = self.prob_density(y)
            pi_k = pi.unsqueeze(dim=2).expand_as(p_k)
            density = torch.sum(
                p_k * pi_k, dim=1
            )
        return density

    def prod_prob(self, y):
        """Taking product of the last dimension of returned probability.
        """
        return torch.prod(self.prob(y), dim=2)


def sample(pi, sigma, mu):
    """Draw samples from a MoG.
    """
    # Choose which gaussian we'll sample from
    pis = Categorical(pi).sample().view(pi.size(0), 1, 1)
    # Choose a random sample, one randn for batch X output dims
    # Do a (output dims)X(batch size) tensor here, so the broadcast works in
    # the next step, but we have to transpose back.
    gaussian_noise = torch.randn(
        (sigma.size(2), sigma.size(0)), requires_grad=False)
    variance_samples = sigma.gather(1, pis).detach().squeeze()
    mean_samples = mu.detach().gather(1, pis).squeeze()
    return (gaussian_noise * variance_samples + mean_samples).transpose(0, 1)
