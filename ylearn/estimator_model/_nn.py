import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Independent, MixtureSameFamily, \
    Normal
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


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
        return self.data[index], self.w[index], self.target[index]


class DiscreteIOBatchData(BatchData):
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
            X = torch.argmax(X, dim=1)
            y = torch.argmax(y, dim=1)

        super(DiscreteIOBatchData, self).__init__(
            X=X, W=W, y=y, X_test=X_test, y_test=y_test, train=train)


class DiscreteIBatchData(BatchData):
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
            X = torch.argmax(X, dim=1)

        super(DiscreteIBatchData, self).__init__(
            X=X, W=W, y=y, X_test=X_test, y_test=y_test, train=train)


class DiscreteOBatchData(BatchData):
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
            y = torch.argmax(y, dim=1)

        super(DiscreteOBatchData, self).__init__(
            X=X, W=W, y=y, X_test=X_test, y_test=y_test, train=train)


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
            -0.5 * ((y - self.mu) / self.sigma) ** 2
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


# def sample(pi, sigma, mu):
#     """Draw samples from a MoG.
#     """
#     # Choose which gaussian we'll sample from
#     pis = Categorical(pi).sample().view(pi.size(0), 1, 1)
#     # Choose a random sample, one randn for batch X output dims
#     # Do a (output dims)X(batch size) tensor here, so the broadcast works in
#     # the next step, but we have to transpose back.
#     gaussian_noise = torch.randn(
#         (sigma.size(2), sigma.size(0)), requires_grad=False
#     )
#     variance_samples = sigma.gather(1, pis).detach().squeeze()
#     mean_samples = mu.detach().gather(1, pis).squeeze()
#     return (gaussian_noise * variance_samples + mean_samples).transpose(0, 1)


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
