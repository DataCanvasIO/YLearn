"""
Adapted from notears.utils.
"""
import numpy as np
import torch
from scipy.special import expit as sigmoid
import igraph as ig
import networkx as nx
import random


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def simulate_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.

    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """

    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm


def simulate_parameter(B, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
    """Simulate SEM parameters for a DAG.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        w_ranges (tuple): disjoint weight ranges

    Returns:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
    """
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W


def simulate_linear_sem(W, n, sem_type, noise_scale=None):
    """Simulate samples from linear SEM with specified type of noise.

    For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        n (int): num of samples, n=inf mimics population risk
        sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
    """

    def _simulate_single_equation(X, w, scale):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if sem_type == 'gauss':
            z = np.random.normal(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=n)
            x = X @ w + z
        elif sem_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        else:
            raise ValueError('unknown sem type')
        return x

    d = W.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or has length d')
        scale_vec = noise_scale
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    if np.isinf(n):  # population risk for linear gauss SEM
        if sem_type == 'gauss':
            # make 1/d X'X = true cov
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            return X
        else:
            raise ValueError('population risk not available')
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
    return X


def simulate_nonlinear_sem(B, n, sem_type, noise_scale=None):
    """Simulate samples from nonlinear SEM.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        n (int): num of samples
        sem_type (str): mlp, mim, gp, gp-add
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix
    """

    def _simulate_single_equation(X, scale):
        """X: [n, num of parents], x: [n]"""
        z = np.random.normal(scale=scale, size=n)
        pa_size = X.shape[1]
        if pa_size == 0:
            return z
        if sem_type == 'mlp':
            hidden = 100
            W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
            W1[np.random.rand(*W1.shape) < 0.5] *= -1
            W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
            W2[np.random.rand(hidden) < 0.5] *= -1
            x = sigmoid(X @ W1) @ W2 + z
        elif sem_type == 'mim':
            w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w1[np.random.rand(pa_size) < 0.5] *= -1
            w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w2[np.random.rand(pa_size) < 0.5] *= -1
            w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w3[np.random.rand(pa_size) < 0.5] *= -1
            x = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z
        elif sem_type == 'gp':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = gp.sample_y(X, random_state=None).flatten() + z
        elif sem_type == 'gp-add':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = sum([gp.sample_y(X[:, i, None], random_state=None).flatten()
                     for i in range(X.shape[1])]) + z
        else:
            raise ValueError('unknown sem type')
        return x

    d = B.shape[0]
    scale_vec = noise_scale if noise_scale else np.ones(d)
    X = np.zeros([n, d])
    G = ig.Graph.Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], scale_vec[j])
    return X

def generate_tri(nodes_dim, graph_type, order, degree, low_value=0.3, high_value=0.5): #num_nodes是变量数
    A_bin = np.zeros((nodes_dim, nodes_dim, order))
    if graph_type == 'ER':
        for p in range(order):
            A_bin[:, :, p] = simulate_random_graph(nodes_dim, graph_type, order, degree,
                                                   Bernoulli_trial=0, is_acyclic=1)
    return A_bin

def simulate_random_graph(nodes_dim, graph_type, order, degree, Bernoulli_trial=0, is_acyclic=1):
    def _random_permutation():
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(nodes_dim))
        return P.T @ A_bin @ P

    if graph_type == 'ER':
        A_bin = simulate_er_dag(degree, nodes_dim, Bernoulli_trial, is_acyclic)
    elif graph_type == 'SF':
        A_bin = simulate_sf_dag(degree, nodes_dim)
    else:
        raise ValueError("Unknown graph type.")
    return _random_permutation()

def simulate_er_dag(degree, nodes_dim, Bernoulli_trial, is_acyclic):

    def _get_acyclic_graph(A_und, is_acyclic):
        return np.tril(A_und, k=is_acyclic)

    def _graph_to_adjmat(G):
        return nx.to_numpy_matrix(G)

    p = float(degree) / (nodes_dim - Bernoulli_trial)
    G_und = nx.generators.erdos_renyi_graph(n=nodes_dim, p=p)
    A_und_bin = _graph_to_adjmat(G_und)  # Undirected
    A_bin = _get_acyclic_graph(A_und_bin, is_acyclic)
    return A_bin

def simulate_sf_dag(degree, nodes_dim):
    m = int(round(degree / 2))
    G = ig.Graph.Barabasi(n=nodes_dim, m=m, directed=True)
    A_bin = np.array(G.get_adjacency().data)
    return A_bin

def simulate_weight_A(A_bin, order, scale=1.0, low=0.3, high=0.5, eta=1.5):
    nums_dim = A_bin.shape[0]
    A = np.zeros((nums_dim, nums_dim, order))
    for k_order in range(1, A.shape[2]+1):
        A_ranges = (
        (scale * -low * (1 / eta ** (k_order - 1)), scale * -high * (1 / eta ** (k_order - 1))),
        (scale * high * (1 / eta ** (k_order - 1)), scale * low * (1 / eta ** (k_order - 1))))
        S = np.random.randint(len(A_ranges), size=(nums_dim, nums_dim))
        for i, (low_A, high_A) in enumerate(A_ranges):
            U = np.random.uniform(low=low_A, high=high_A, size=(nums_dim, nums_dim))
            A[:, :, k_order-1] += A_bin[:, :, k_order-1] * (S == i) * U
    return A


def simulate_linear_sem_with_A(A, B, lagX, n, sem_type, noise_scale=None):
    def _simulate_single_equation(X, B, scale, time):
        if sem_type == 'gauss':
            z = np.random.normal(scale=scale, size=n)
            # print(X.shape, B.shape, z.shape)
            x = X @ B + time * z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            x = X @ B + time * z
        elif sem_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=n)
            x = X @ B + time * z
        elif sem_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=n)
            x = X @ B + time * z
        elif sem_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ B)) * 1.0
        elif sem_type == 'poisson':
            x = np.random.poisson(np.exp(X @ B)) * 1.0
        else:
            raise ValueError('unknown sem type')
        return x
    d = B.shape[0]
    order = A.shape[2]
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or has length d')
        scale_vec = noise_scale
    if not is_dag(B):
        raise ValueError('B must be a DAG')
    if np.isinf(n):  # population risk for linear gauss SEM
        if sem_type == 'gauss':
            # make 1/d X'X = true cov
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - B)
            return X
        else:
            raise ValueError('population risk not available')

    G = ig.Graph.Weighted_Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n,d])
    for j in ordered_vertices:
        intra_parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, intra_parents], B[intra_parents, j], scale_vec[j], 0)
        for k in range(order):
            inter_G = ig.Graph.Weighted_Adjacency(A[:, :, k].tolist())
            inter_parents = inter_G.neighbors(j, mode=ig.IN)
            X[:, j] += _simulate_single_equation(lagX[k, :, inter_parents].T, A[inter_parents, j, k], scale_vec[j], 1)
    return X

def count_accuracy(B_true, B_est):
    """Compute various accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    # if (B_est == -1).any():  # cpdag
    #     if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
    #         raise ValueError('B_est should take value in {0,1,-1}')
    #     if ((B_est == -1) & (B_est.T == -1)).any():
    #         raise ValueError('undirected edge should only appear once')
    # else:  # dag
    #     if not ((B_est == 0) | (B_est == 1)).all():
    #         raise ValueError('B_est should take value in {0,1}')
    #     if not is_dag(B_est):
    #         raise ValueError('B_est should be a DAG')
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size}


def gen(n=100, d=5, s0=5, noise_scale=None):
    s0 = 1 * d
    graph_type, sem_type = 'ER', 'gauss'
    B_true = simulate_dag(d, s0, graph_type)
    W_true = simulate_parameter(B_true)
    return simulate_linear_sem(W_true, n, sem_type, noise_scale=noise_scale)


def dygen(n=5, d=3, step=5, order=2, noise_scale=None):
    """
    Adapted from GraphNotears.
    """
    assert order <= step
    s0 = 1 * d
    degree = 2
    graph_type, sem_type = 'ER', 'gauss'
    B_bin = simulate_dag(d, s0, graph_type)
    B_mat = simulate_parameter(B_bin)

    A_bin = generate_tri(d, graph_type, order, degree)
    A_mat = simulate_weight_A(A_bin, order)

    Xbase = np.array([simulate_linear_sem(B_mat, n, sem_type, noise_scale=noise_scale) for _ in range(order)])
    # 生成后续的Xbase
    for i in range(step):
        Xbase1 = simulate_linear_sem_with_A(A_mat, B_mat, Xbase[-order:], n, sem_type, noise_scale=noise_scale)
        Xbase = np.append(Xbase, [Xbase1], axis=0)
    Xbase = np.array(Xbase)
    Xbase = Xbase.reshape(-1, Xbase.shape[2])
    return Xbase, B_mat, A_mat


if __name__ == '__main__':
    X = dygen()
    print(X.shape)
