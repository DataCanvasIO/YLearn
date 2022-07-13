import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def cartesian(arrays):
    n = len(arrays)
    cart_prod = np.array(np.meshgrid(*arrays)).T.reshape(-1, n)
    return cart_prod


def _get_wv(w, v):
    if w is None:
        wv = v
    else:
        if v is not None:
            wv = np.concatenate((w, v), axis=1)
        else:
            wv = w

    return wv


def get_wv(*wv):
    return np.concatenate([w for w in wv if w is not None], axis=1)


def get_tr_ctrl(tr_crtl, trans, *, treat=False, one_hot=False, discrete_treat=True):
    if tr_crtl is None or not discrete_treat:
        tr_crtl = 1 if treat else 0
    else:
        if not isinstance(tr_crtl, np.ndarray):
            if not isinstance(tr_crtl, (list, tuple)):
                tr_crtl = [tr_crtl]
            tr_crtl = np.array(tr_crtl).reshape(1, -1)

        tr_crtl = trans(tr_crtl).reshape(1, -1)

        if not one_hot:
            tr_crtl = convert4onehot(tr_crtl).astype(int)[0]

    return tr_crtl


def get_treat_control(treat_ctrl, trans, treat=False):
    n_treat = len(trans.categories_)

    if treat_ctrl is not None:
        if not isinstance(treat_ctrl, int):
            assert len(treat_ctrl) == n_treat

        treat_ctrl = np.array(list(treat_ctrl))
        treat_ctrl = trans.transform(treat_ctrl.reshape(1, -1))
    else:
        if treat:
            treat_ctrl = np.ones((1, n_treat)).astype(int)
        else:
            treat_ctrl = np.zeros((1, n_treat)).astype(int)

    return treat_ctrl


def shapes(*tensors, all_dim=False):
    shapes = []
    if all_dim:
        for tensor in tensors:
            if tensor is not None:
                shapes.append(tensor.shape)
    else:
        for tensor in tensors:
            if tensor is not None:
                shapes.append(tensor.shape[1])

    return shapes


def nd_kron(x, y):
    assert x.shape[0] == y.shape[0]
    fn = np.vectorize(np.kron, signature='(n),(m)->(k)')
    kron_prod = fn(x, y)

    return kron_prod


def tensor_or_none(x):
    if x is not None:
        import torch
        return torch.tensor(x)
    else:
        return None


def convert2tensor(*arrays):
    # arrays = list(arrays)
    # for i, array in enumerate(arrays):
    #     if array is not None:
    #         arrays[i] = torch.tensor(array)
    #
    # return arrays
    return tuple(map(tensor_or_none, arrays))


def convert4onehot(x):
    return np.dot(x, np.arange(0, x.shape[1]).T)


def get_groups(target, a, one_hot, *arrays):
    arrays = list(arrays)
    
    if one_hot:
        a = convert4onehot(a)
        label = (a == target)
    # label = np.all(a == target, axis=1)
    else:
        label = np.all(a == target, axis=1)

    for i, array in enumerate(arrays):
        arrays[i] = array[label]

    return arrays


def convert2array(data, *S, tensor=False):
    assert isinstance(data, pd.DataFrame)

    def _get_array(cols):
        if cols is not None:
            r = data[cols].values
            if len(r.shape) == 1:
                r = np.expand_dims(r, axis=1)
        else:
            r = None
        return r

    S = map(_get_array, S)

    if tensor:
        S = map(tensor_or_none, S)

    return tuple(S)


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

#
# class DiscreteIOBatchData(Dataset):
#     def __init__(
#         self,
#         X=None,
#         W=None,
#         y=None,
#         X_test=None,
#         y_test=None,
#         train=True,
#     ):
#         if train:
#             self.w = W
#             self.data = torch.argmax(X, dim=1)
#             self.target = torch.argmax(y, dim=1)
#         else:
#             self.w = W
#             self.data = X_test
#             self.target = y_test
#
#     def __len__(self):
#         return self.target.shape[0]
#
#     def __getitem__(self, index):
#         return self.data[index], self.w[index, :], self.target[index]
#
#
# class DiscreteIBatchData(Dataset):
#     def __init__(
#         self,
#         X=None,
#         W=None,
#         y=None,
#         X_test=None,
#         y_test=None,
#         train=True,
#     ):
#         if train:
#             self.w = W
#             self.data = torch.argmax(X, dim=1)
#             self.target = y
#         else:
#             self.w = W
#             self.data = X_test
#             self.target = y_test
#
#     def __len__(self):
#         return self.target.shape[0]
#
#     def __getitem__(self, index):
#         return self.data[index], self.w[index, :], self.target[index, :]
#
#
# class DiscreteOBatchData(Dataset):
#     def __init__(
#         self,
#         X=None,
#         W=None,
#         y=None,
#         X_test=None,
#         y_test=None,
#         train=True,
#     ):
#         if train:
#             self.w = W
#             self.data = X
#             self.target = torch.argmax(y, dim=1)
#         else:
#             self.w = W
#             self.data = X_test
#             self.target = y_test
#
#     def __len__(self):
#         return self.target.shape[0]
#
#     def __getitem__(self, index):
#         return self.data[index, :], self.w[index, :], self.target[index]
