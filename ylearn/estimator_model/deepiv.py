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

from torch.autograd import Variable
from copy import deepcopy
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader
from torch.distributions import Categorical, Normal, MixtureSameFamily,\
    Independent

from .utils import convert2array, convert2tensor, shapes
from ._nn import BatchData, DiscreteOBatchData, DiscreteIOBatchData, DiscreteIBatchData
from .base_models import BaseEstModel

# We first build the mixture density network.


class MixtureDensityNetwork(nn.Module):
    """
    See (https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf) for
    reference.
    """

    def __init__(
        self, z_d, w_d, out_d,
        hidden_d1=128,
        hidden_d2=64,
        hidden_d3=32,
        num_gaussian=5,
        is_discrete_input=False,
        is_discrete_output=False,
        embedding_dim=None,
        stab_bound=200
    ):
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

        self.stab_bound = stab_bound
        self._z_d = z_d
        self._w_d = w_d
        self._out_d = out_d
        self.num_gaussian = num_gaussian
        self.is_discrete_input = is_discrete_input

        if is_discrete_input:
            assert z_d is not None, 'Please specify the dimension of the'
            'treatment vector.'

            embedding_dim = z_d if embedding_dim is None else embedding_dim
            self.embed = nn.Embedding(z_d, embedding_dim)
            in_d = int(embedding_dim + w_d)
        else:
            self.embed = nn.Identity()
            in_d = int(z_d + w_d)

        self.hidden_layer = nn.Sequential(
            nn.Linear(in_d, hidden_d1),
            nn.ELU(),
            nn.Linear(hidden_d1, hidden_d2),
            nn.ELU(),
        )

        self.pi = nn.Linear(hidden_d2, num_gaussian)
        self.sigma = nn.Linear(hidden_d2, num_gaussian * out_d)
        self.mu = nn.Linear(hidden_d2, num_gaussian * out_d)

    def forward(self, x, w):
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
        x = self.embed(x)
        # x = torch.cat((x, w), dim=1).to(torch.float32)
        x = torch.cat((x, w), dim=1)
        h = self.hidden_layer(x)
        # h += 1 + 1e-15
        pi = self.pi(h)
        pi = pi - pi.max(dim=1).values.reshape(-1, 1)
        pi = nn.Softmax(dim=1)(pi)
        # pi = F.gumbel_softmax(pi, hard=True, dim=1)
        mu = self.mu(h).reshape(
            -1, self.num_gaussian, self._out_d
        )
        sigma = torch.clamp(self.sigma(h), -self.stab_bound, self.stab_bound)
        sigma = torch.exp(sigma).reshape(-1, self.num_gaussian, self._out_d)
        return pi, mu, sigma

# To make the above MDN consistant to the standard machine learning models, we
# use the following wrapper to wrap it so that the methods such as fit() can
# be applied.


class MDNWrapper:
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

        self.model = deepcopy(mdn)
        self._z_d = mdn._z_d
        self._w_d = mdn._w_d
        self._out_d = mdn._out_d
        self.num_gaussian = mdn.num_gaussian
        self.is_discrete_input = mdn.is_discrete_input

    def mg(self, pi, mu, sigma):
        mix = Categorical(pi)
        comp = Independent(Normal(mu, sigma), 1)
        mg = MixtureSameFamily(mix, comp)
        return mg

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
        mg = self.mg(pi, mu, sigma)
        loss = mg.log_prob(y)
        return -torch.logsumexp(loss, dim=0)

    def fit(
        self, z, w,
        target,
        device='cuda',
        lr=0.01,
        epoch=100,
        optimizer='SGD',
        batch_size=64,
        **optim_config
    ):
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

        if w is None:
            w = z[:, -1:-1]

        if self.is_discrete_input:
            data = DiscreteIBatchData(X=z, W=w, y=target)
        else:
            data = BatchData(X=z, W=w, y=target)
        train_loader = DataLoader(data, batch_size=batch_size)

        op_dict = {
            'SGD': optim.SGD(self.model.parameters(), lr=lr),
            'Adam': optim.Adam(self.model.parameters(), lr=lr, **optim_config)
        }
        optimizer = op_dict[optimizer]

        for e in range(epoch):
            for i, (z, w, y) in enumerate(train_loader):
                self.model.train()
                z, w, y = z.to(device), w.to(device), y.to(device)
                pi, mu, sigma = self.model(z, w)
                loss = self.loss_fn(pi, mu, sigma, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'Finished {e+1}/{epoch} epochs | current loss {loss.data}')

    def predict_log_prob(self, z, w, target):
        """Calculate the probability density P(y|X) with the trained mixture density
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
        pi, mu, sigma = self.model(z, w)
        log_prob = self.mg(pi, mu, sigma).log_prob(target)
        return log_prob

    def predict_prob(self, z, w, target):
        return torch.exp(self.predict_log_prob(z, w, target))

    def predict_cdf(self, z, w, target):
        pi, mu, sigma = self.model(z, w)
        return self.mg(pi, mu, sigma).cdf(target)

    def _sample(self, z, w, sample_n):
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
        pi, mu, sigma = self.model(z, w)
        mg = self.mg(pi, mu, sigma)
        sampled = mg.sample((sample_n, ))
        return sampled.reshape(-1, self._out_d)


class Net(nn.Module):
    # This neural network is for x_net with discrete x (treatment) and all
    # y_net. If x is continuous, use MixtureDensityNetwork.
    def __init__(self, x_d, w_d, out_d,
                 hidden_d1=256,
                 hidden_d2=512,
                 hidden_d3=256,
                 is_discrete_input=False,
                 is_discrete_output=False,
                 embedding_dim=None):
        super().__init__()

        self._x_d = x_d
        self._out_d = out_d
        self.is_discrete_input = is_discrete_input
        self.is_discrete_output = is_discrete_output

        if is_discrete_input:
            assert x_d is not None, 'Please specify the dimension of the'
            'treatment vector.'

            embedding_dim = x_d if embedding_dim is None else embedding_dim
            self.embed = nn.Embedding(x_d, embedding_dim)
            in_d = int(embedding_dim + w_d)
        else:
            self.embed = nn.Identity()
            in_d = int(x_d + w_d)

        self.fc1 = nn.Linear(in_d, hidden_d1)
        self.fc2 = nn.Linear(hidden_d1, hidden_d2)
        self.fc3 = nn.Linear(hidden_d2, hidden_d3)
        self.fc4 = nn.Linear(hidden_d3, out_d)

    def forward(self, x, w):
        # this definition should be simplified
        # For discrete treatment, x should be a vector of labels not one-hot
        # tensors
        x = self.embed(x)
        x = torch.cat((x, w), dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        output = self.fc4(x)
        return output


class NetWrapper:
    # This wrapper is for x_net with discrete x (treatment) and all y_net
    # If x is continuous, use MDNWrapper.
    def __init__(self, net, is_y_net=False):
        self.model = deepcopy(net)
        self.is_y_net = is_y_net
        self.is_discrete_input = net.is_discrete_input
        self.is_discrete_output = net.is_discrete_output
        self._x_d = net._x_d
        self._out_d = net._out_d

    def fit(
        self, x, w,
        target,
        device='cuda',
        lr=0.01,
        epoch=500,
        optimizer='SGD',
        batch_size=64,
        **optim_config
    ):
        # TODO:tqdm
        """if is_discrete_input: transform x
        if is_discrete_output: default loss should be nn.NLLLoss


        Args:
            x (_type_): _description_
            w (_type_): _description_
            target (_type_): _description_
            device (str, optional): _description_. Defaults to 'cuda'.
            lr (float, optional): _description_. Defaults to 0.01.
            epoch (int, optional): _description_. Defaults to 500.
            optimizer (str, optional): _description_. Defaults to 'SGD'.
            batch_size (int, optional): _description_. Defaults to 64.
        """
        # move model to device
        self.model = self.model.to(device)

        # convert w to tensor([]) with shape (x.shape[0], 0) if w is None
        if w is None:
            w = x[:, -1:-1]

        # use different data set for efficiency
        if self.is_discrete_output:
            if self.is_discrete_input and not self.is_y_net:
                data = DiscreteIOBatchData(X=x, W=w, y=target)
            else:
                data = DiscreteOBatchData(X=x, W=w, y=target)

            loss_fn = optim_config.pop('loss', nn.CrossEntropyLoss())
        else:
            if self.is_discrete_input and not self.is_y_net:
                data = DiscreteIBatchData(X=x, W=w, y=target)
            else:
                data = BatchData(X=x, W=w, y=target)

            loss_fn = optim_config.pop('loss', nn.MSELoss())

        # make batched data
        train_loader = DataLoader(data, batch_size=batch_size)

        op_dict = {
            'SGD': optim.SGD(self.model.parameters(), lr=lr),
            'Adam': optim.Adam(self.model.parameters(), lr=lr, **optim_config)
        }
        # prepare for training
        optimizer = op_dict[optimizer]

        if self.is_discrete_input and self.is_y_net:
            x_d = self._x_d
            out_d = self._out_d
            x_label = torch.arange(x_d)
            for e in range(epoch):
                for i, (X, W, y) in enumerate(train_loader):
                    self.model.train()
                    X, W, y = X.to(device), W.to(device), y.to(device)

                    x = Variable(X)  # TODO: I dont konw why this works, why???
                    batch_num = y.shape[0]
                    # after sampled, both batches of w_sample and x_sample
                    # will have (batch_num*x_d) samples
                    w_sample = W.repeat_interleave(x_d, dim=0)
                    x_sample = x_label.repeat(batch_num, )

                    # y_pred_vec has shape (batch_num*x_d, out_d)
                    y_pred = self.model(x_sample, w_sample)
                    y_pred = y_pred.reshape(batch_num, x_d, out_d)
                    y_pred = torch.einsum('bkj,bk->bj', [y_pred, x])
                    loss = loss_fn(y_pred, y)

                    optimizer.zero_grad()
                    # loss.backward(retain_graph=True)
                    loss.backward()
                    optimizer.step()
                print(
                    f'Finished {e+1}/{epoch} epochs | current loss {loss.data}'
                )
        else:
            for e in range(epoch):
                for i, (X, W, y) in enumerate(train_loader):
                    self.model.train()
                    X, W, y = X.to(device), W.to(device), y.to(device)
                    y_pred = self.model(X, W)
                    loss = loss_fn(y_pred, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                print(
                    f'Finished {e+1}/{epoch} epochs | current loss {loss.data}'
                )

    def predict_proba(self, x, w):
        assert self.is_discrete_output, 'Please use predict(x, w) for'
        'continous output.'
        if self.is_discrete_input:
            x = torch.argmax(x, dim=1)

        pred = self.model(x, w)
        return nn.Softmax(dim=1)(pred)

    def predict(self, x, w):
        if self.is_discrete_input:
            x = torch.argmax(x, dim=1)

        if self.is_discrete_output:
            pred = nn.Softmax(dim=1)(self.model(x, w))
            return torch.argmax(pred, dim=1)
        else:
            return self.model(x, w)

    def _sample(self, x, w, sample_n):
        return self.predict_proba(x, w)


# We are now ready to build the complete model with above wrapped outcome and
# treatment nets.


class DeepIV(BaseEstModel):
    r"""Training of a DeepIV model g(x, w) is composed of 2 stages:
            1. In the first stage, we train a neural network to estimate the
            distribution of the treatment x given the instrument z and
            adjustment (also covariate) w;
            2. In the second stage, we train another neural network to estiamte
            the outcome y givn treatment x and adjustment (also
            covariate) w.
        The trained model is used to estimate the causal effect
            g(x_1, w) - g(x_0, w)
        or
            \partial_x g(x, w).

    Attributes
    ----------
    x_net_kwargs : dict

    y_net_kwargs : dict

    _x_net_init : Net
        The model for the treatment.

    _y_net_init : Net
        The model for the outcome.

    x_net : NetWrapper, optional
        Wrapped treatment model.

    y_net : NetWrapper, optional
        Wrapped outcome model.

    is_discrete_instrument : bool
        True if the instrument is discrete.

    randome_state : int

    is_discrete_treatment : bool

    is_discrete_outcome : bool

    categories : str, optional. Defaults to 'auto'

    x_transformer : OneHotEncoder, optional
        Transformer of the treatment.

    _z_transformer : OneHotEncoder, optional
        Transformer of the instrument

    y_transformer : OneHotEncoder, optional
        Transformer of the outcome

    w : ndarray with shape (n, w_d)
        Adjustment variables in the training data where n is the number of
        examples and w_d is the number of adjustment and covaraite.

    _y_d : ndarray
        Shapes of the outcome variables in the training data.

    _x_d : ndarray
        Shapes of the treatment variables in the training data.

    _w_d : ndarray
        Shapes of the adjustment variables in the training data.

    _z_d : ndarray
        Shapes of the instrumental variables in the training data.

    Methods
    ----------
    fit(data, outcome, treatment, instrument=None, adjustment=None,
        approx_grad=True, sample_n=None, x_net_config=None, 
        y_net_config=None, **kwargs)
        Fit the instance of DeepIV.

    estimate(data=None, treat=None, control=None, quantity='CATE', marginal_effect=False,
             *args, **kwargs,)
        Estimate the causal effect.

    _gen_x_model(x_model, *args, **kwargs)

    _gen_y_model(y_model, *args, **kwargs)

    _prepare4est(data, treat=None, control=None, marginal_effect=False)

    """

    def __init__(
        self,
        x_net=None,
        y_net=None,
        x_hidden_d=None,
        y_hidden_d=None,
        num_gaussian=5,
        is_discrete_treatment=False,
        is_discrete_outcome=False,
        is_discrete_instrument=False,
        categories='auto',
        random_state=2022,
    ):
        """
        Parameters
        ----------
        x_net : ylearn.estimator_model.deepiv.Net
            Representation of the mixture density network for continuous
            treatment or an usual classification net for discrete treatment. If None, the default neural network will be used. See :py:class:`ylearn.estimator_model.deepiv.Net` for reference.

        y_net :  ylearn.estimator_model.deepiv.Net
            Representation of the outcome network. If None, the default neural network will be used. See :py:class:`ylearn.estimator_model.deepiv.Net` for reference.

        x_hidden_d : int, optional. Defaults to None
            Dimension of the hidden layer of the default x_net of DeepIV.

        y_hidden_d : int, optional. Defaults to None
            Dimension of the hidden layer of the default y_net of DeepIV.

        num_gaussian : int, optional. Defaults to 5.
            Number of gaussians when using the mixture density network which will be directly ignored when the treatment is discrete.

        is_discrete_treatment : bool, optional. Defaults to False.

        is_discrete_outcome : bool, optional. Defaults to False.

        is_discrete_instrument : bool, optional. Defaults to False.

        categories : str, optional. Defaults to 'auto'

        random_state : int, optional. Defaults to 2022.
        """
        self.x_net_kwargs = {}
        if x_hidden_d is not None:
            self.x_net_kwargs['hidden_d'] = x_hidden_d
        if num_gaussian is not None and not is_discrete_treatment:
            self.x_net_kwargs['num_gaussian'] = num_gaussian

        self.y_net_kwargs = {}
        if y_hidden_d is not None:
            self.y_net_kwargs['hidden_d'] = y_hidden_d

        self._x_net_init = x_net
        self._y_net_init = y_net
        self.is_discrete_instrument = is_discrete_instrument

        super().__init__(
            random_state=random_state,
            is_discrete_treatment=is_discrete_treatment,
            is_discrete_outcome=is_discrete_outcome,
            # is_discrete_instrument=is_discrete_instrument,
            categories=categories,
        )

    def fit(
        self,
        data,
        outcome,
        treatment,
        instrument=None,
        adjustment=None,
        approx_grad=True,
        sample_n=None,
        x_net_config=None,
        y_net_config=None,
        **kwargs
    ):
        """Train the DeepIV model. #TODO: consider implementing the comp_transformer for multiple treatment

        Parameters
        ----------
        data : DataFrame
            Training dataset for training the estimator.
        outcome : list of str, optional
            Names of the outcome
        treatment : list of str, optional
            Names of the treatment
        instrument : list of str, optional
            Names of the instrument, by default None
        adjustment : list of str, optional
            Names of the adjustment set. #TODO: Note that in the current version we aslo view all adjustment variables as the covariates., by default None
        approx_grad : bool, optional
            Whether use the approximated gradient as in the reference, by default True
        sample_n : int, optional
            Times of new samples when using the approx_grad technique, by default None
        x_net_config : dict, optional
            Configuration of the x_net, by default None
        y_net_config : dict, optional
            Configuration of the y_net, by default None

        Returns
        -------
        instance of DeepIV
            The trained DeepIV model
        """
        assert instrument is not None, 'instrument is required.'

        super().fit(data, outcome, treatment,
                    adjustment=adjustment,
                    instrument=instrument,
                    )

        if x_net_config is None and y_net_config is None:
            x_net_config = kwargs
            y_net_config = kwargs

        y, x, z, w = convert2array(
            data, outcome, treatment, instrument, adjustment
        )

        # transformers for building one hot vectors
        if self.is_discrete_treatment:
            if self.categories == 'auto' or self.categories is None:
                categories = 'auto'
            else:
                categories = list(self.categories)

            self.x_transformer = OneHotEncoder(categories=categories)
            self.x_transformer.fit(x)
            x = self.x_transformer.transform(x).toarray()

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
        self._y_d, self._x_d, self._w_d, self._z_d = shapes(
            y, x, w, z, all_dim=False
        )

        # build networks
        self.x_net = self._gen_x_model(
            self._x_net_init,
            self._z_d,
            self._w_d,
            self._x_d,
            is_discrete_input=self.is_discrete_instrument,
            is_discrete_output=self.is_discrete_treatment,
            **self.x_net_kwargs
        )
        self.y_net = self._gen_y_model(
            self._y_net_init,
            self._x_d,
            self._w_d,
            self._y_d,
            is_discrete_input=self.is_discrete_treatment,
            is_discrete_output=self.is_discrete_outcome,
            **self.y_net_kwargs
        )

        # Step 1: train the model for estimating the treatment given the
        # instrument and adjustment
        self.x_net.fit(z, w, target=x, **x_net_config)

        # Step 2: generate new samples if calculating grad approximately
        if approx_grad:
            x_sampled = self.x_net._sample(z, w, sample_n)
        else:
            # TODO: the loss funcn should be modified if not approx_grad
            x_sampled = x

        # Step 3: fit the final counterfactual prediction model
        if self.is_discrete_treatment:
            w_sampled = w
        else:
            w_sampled = w.repeat(sample_n, 1) if w is not None else None

        # TODO: be careful here
        self.y_net.fit(x_sampled, w_sampled, target=y, **y_net_config)

        self._is_fitted = True
        return self

    def effect_nji(self, data=None):
        if not hasattr(self, 'x_net') or not hasattr(self, 'y_net'):
            raise Exception('The estimator is not fitted yet.')

        if data is None:
            w = self.w
        else:
            w = convert2tensor(
                convert2array(data, self.adjustment)[0]
            )[0]

        n = w.shape[0]
        ones = torch.eye(n, self._x_d)

        y_nji = torch.full((n, self._y_d, self._x_d))

        if self.is_discrete_treatment:
            for treat in range(self._x_d):
                treat_id = (torch.ones(n, ) * treat).int()

                xt = ones.index_select(dim=0, index=treat_id)
                y_pred = self.y_net.predict(xt, w)
                y_nji[:, :, treat] = y_pred.reshape(n, self._y_d)

            y_ctrl = y_nji[:, :, 0].reshape(n, -1, 1).repeat((1, 1, self._x_d))
        else:
            xt = ones * 1
            x0 = ones * 0
            y_nji[:, :, 0] = self.y_net.predict(x0, w)
            y_nji[:, :, 1] = self.y_net.predict(xt, w)
            y_ctrl = y_nji[:, :, 0].reshape(n, -1, 1).repeat((1, 1, 2))

        y_nji = y_nji - y_ctrl

        return y_nji

    def _prepare4est(
        self,
        data=None,
        treat=None,
        control=None,
        marginal_effect=False,
        *args,
        **kwargs
    ):
        if not hasattr(self, 'x_net') or not hasattr(self, 'y_net'):
            raise Exception('The estimator is not fitted yet.')

        treat = 1 if treat is None else treat
        control = 0 if control is None else control

        if data is None:
            w = self.w
        else:
            w = convert2tensor(
                convert2array(data, self.adjustment)[0]
            )[0]
        n = w.shape[0]
        ones = torch.eye(n, self._x_d)

        if self.is_discrete_treatment:
            # build the one_hot vector for treatment vector
            treat_id = (torch.ones(n, ) * treat).int()

            # build treatment vector xt and control vector x0
            xt = ones.index_select(dim=0, index=treat_id)
            x0 = ones.index_select(dim=0, index=torch.zeros(n, ).int())

            return (self.y_net.predict(xt, w), self.y_net.predict(x0, w))
        else:
            xt = ones * treat
            x0 = ones * control
            xt.requires_grad = True

            yt = self.y_net.predict(xt, w)

            if marginal_effect:
                return (xt.grad.detach(), )
            else:
                y0 = self.y_net.predict(x0, w)
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
        #TODO: update the definition of treat and control.
        """Estimate the causal effect with the type of the quantity.

        Parameters
        ----------
        data : DataFrame, optional
            Test data. The model will use the training data if None, by default None
        treat : int, optional
            Value of the treatment, by default None. If None, then the model will set treat=1.
        control : int, optional
            Value of the control, by default None. If None, then the model will set control=0.
        quantity : str, optional
            Option for returned estimation result. The possible values of quantity include:
                
                1. *'CATE'* : the estimator will evaluate the CATE;
                
                2. *'ATE'* : the estimator will evaluate the ATE;
                
                3. *None* : the estimator will evaluate the ITE or CITE.
        marginal_effect : bool, optional
            _description_, by default False

        Returns
        -------
        torch.tensor
            Estimated causal effects
        """
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
        # elif quantity == 'Counterfactual prediction':
        else:
            return yt

    def _gen_x_model(self, x_model, *args, **kwargs):
        hidden_d_list = kwargs.pop('hidden_d', [64, 128, 64])
        for i, hidden_d in enumerate(hidden_d_list, 1):
            kwargs[f'hidden_d{i}'] = hidden_d

        if self.is_discrete_treatment:
            if x_model is None:
                assert any(args), 'Need parameters to define a treatment net.'
                x_model = Net(*args, **kwargs)

            x_net = NetWrapper(x_model, is_y_net=False)
        else:
            if x_model is None:
                assert any(args), 'Need parameters to define a treatment net.'
                x_model = MixtureDensityNetwork(*args, **kwargs)

            x_net = MDNWrapper(x_model)

        return x_net

    def _gen_y_model(self, y_model, *args, **kwargs):
        hidden_d_list = kwargs.pop('hidden_d', [64, 128, 64])
        for i, hidden_d in enumerate(hidden_d_list, 1):
            kwargs[f'hidden_d{i}'] = hidden_d

        if y_model is None:
            assert any(args), 'Need parameters to define an outcome net.'
            y_net = Net(*args, **kwargs)

        return NetWrapper(y_net, is_y_net=True)
