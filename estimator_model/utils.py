import re
import torch
import math

import torch.nn as nn
import numpy as np

from torch.distributions import Categorical, Independent, MixtureSameFamily, \
    Normal
from torch.utils.data import Dataset

from sklearn.preprocessing import OneHotEncoder


def shapes(*tensors, all_dim=False):
    shapes = [None for i in range(len(tensors))]
    if all_dim:
        for i, tensor in enumerate(tensors):
            if tensor is not None:
                shapes[i] = tensor.shape
    else:
        for i, tensor in enumerate(tensors):
            if tensor is not None:
                shapes[i] = tensor.shape[1]

    return shapes


def nd_kron(x, y):
    dim = x.shape[0]
    assert dim == y.shape[0]
    kron_prod = np.kron(x[0], y[0]).reshape(1, -1)

    if dim > 1:
        for i, vec in enumerate(x[1:], 1):
            kron_prod = np.concatenate(
                (kron_prod, np.kron(vec, y[i]).reshape(1, -1)), axis=0
            )

    return kron_prod


def convert2tensor(*arrays):
    arrays = list(arrays)
    for i, array in enumerate(arrays):
        if array is not None:
            arrays[i] = torch.tensor(array)

    return arrays


def convert4onehot(x):
    return np.dot(x, np.arange(0, x.shape[1]).T)


def convert2array(*S, tensor=False):
    data = S[0]
    S = list(S[1:])

    for i, s in enumerate(S):
        if s is not None:
            si = data[s].values
            if len(si.shape) == 1:
                si = np.expand_dims(si, axis=1)
        else:
            si = None

        S[i] = si

    if tensor:
        for si in S:
            S[i] = torch.tensor(si)

    return S


def convert2str(*S):
    S = list(S)
    for i, s in enumerate(S):
        if isinstance(s, str):
            S[i] = tuple(s)
    return S


def one_hot_transformer(*S):
    transformer_list = []

    for s in S:
        if s[0]:
            temp_transormer = OneHotEncoder()
            temp_transormer.fit(s[1])
        else:
            temp_transormer = None

        transformer_list.append(temp_transormer)

    return transformer_list


class DiscreteIOBatchData(Dataset):
    def __init__(
        self,
        X=None,
        W=None,
        y=None,
        X_test=None,
        y_test=None,
        train=True,
    ):
        if train:
            self.w = W
            self.data = torch.argmax(X, dim=1)
            self.target = torch.argmax(y, dim=1)
        else:
            self.w = W
            self.data = X_test
            self.target = y_test

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.w[index, :], self.target[index]


class DiscreteIBatchData(Dataset):
    def __init__(
        self,
        X=None,
        W=None,
        y=None,
        X_test=None,
        y_test=None,
        train=True,
    ):
        if train:
            self.w = W
            self.data = torch.argmax(X, dim=1)
            self.target = y
        else:
            self.w = W
            self.data = X_test
            self.target = y_test

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.w[index, :], self.target[index, :]


class DiscreteOBatchData(Dataset):
    def __init__(
        self,
        X=None,
        W=None,
        y=None,
        X_test=None,
        y_test=None,
        train=True,
    ):
        if train:
            self.w = W
            self.data = X
            self.target = torch.argmax(y, dim=1)
        else:
            self.w = W
            self.data = X_test
            self.target = y_test

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, index):
        return self.data[index, :], self.w[index, :], self.target[index]


class BatchData(Dataset):
    def __init__(
        self,
        X=None,
        W=None,
        y=None,
        X_test=None,
        y_test=None,
        train=True,
    ):
        if train:
            self.w = W
            self.data = X
            self.target = y
        else:
            self.w = W
            self.data = X_test
            self.target = y_test

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, index):
        return self.data[index, :], self.w[index, :], self.target[index, :]


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
            density = torch.sum(p_k * pi_k, dim=1)
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
        (sigma.size(2), sigma.size(0)), requires_grad=False
    )
    variance_samples = sigma.gather(1, pis).detach().squeeze()
    mean_samples = mu.detach().gather(1, pis).squeeze()
    return (gaussian_noise * variance_samples + mean_samples).transpose(0, 1)
