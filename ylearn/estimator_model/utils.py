import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from numpy.linalg import inv

from joblib import effective_n_jobs


def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(effective_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = np.full(n_jobs, n_estimators // n_jobs, dtype=int)
    n_estimators_per_job[: n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()


def count_leaf_num(y_train):
    """y_train is the predicted outcome of the tree. Currently we assume it as 1-dimensional.

    Parameters
    ----------
    y_train : np.ndarray
    """
    y_train = y_train.reshape(-1, 1)
    value, num = np.unique(y_train, return_counts=True)
    num = np.tile(num.reshape(1, -1), (y_train.shape[0], 1))
    return num[y_train == value].reshape(-1, 1)


def inverse_grad(grad, eps=1e-5):
    Id = np.eye(grad.shape[-1]) * eps
    if grad.ndim > 2:
        Id = np.tile(Id, grad.shape[:-2] + (1, 1))
        grad += Id
    else:
        grad += Id
    return inv(grad)


def check_classes(target, classes_):
    if target is None:
        return None
    else:
        assert classes_ is not None
        assert target in classes_
        return np.where(target == classes_)[0]


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
    if tr_crtl is None:
        return 1 if treat else 0

    if discrete_treat:
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
        else:
            treat_ctrl = [
                treat_ctrl,
            ]

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
    fn = np.vectorize(np.kron, signature="(n),(m)->(k)")
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
        label = a == target
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


"""The following two functions are forked from sklearn
"""


# def make_constraint(constraint):
#     """Convert the constraint into the appropriate Constraint object.
#     Parameters
#     ----------
#     constraint : object
#         The constraint to convert.
#     Returns
#     -------
#     constraint : instance of _Constraint
#         The converted constraint.
#     """
#     if isinstance(constraint, str) and constraint == "array-like":
#         return _ArrayLikes()
#     if isinstance(constraint, str) and constraint == "sparse matrix":
#         return _SparseMatrices()
#     if isinstance(constraint, str) and constraint == "random_state":
#         return _RandomStates()
#     if constraint is callable:
#         return _Callables()
#     if constraint is None:
#         return _NoneConstraint()
#     if isinstance(constraint, type):
#         return _InstancesOf(constraint)
#     if isinstance(constraint, (Interval, StrOptions, Options, HasMethods)):
#         return constraint
#     if isinstance(constraint, str) and constraint == "boolean":
#         return _Booleans()
#     if isinstance(constraint, str) and constraint == "verbose":
#         return _VerboseHelper()
#     if isinstance(constraint, str) and constraint == "missing_values":
#         return _MissingValues()
#     if isinstance(constraint, str) and constraint == "cv_object":
#         return _CVObjects()
#     if isinstance(constraint, Hidden):
#         constraint = make_constraint(constraint.constraint)
#         constraint.hidden = True
#         return constraint
#     raise ValueError(f"Unknown constraint type: {constraint}")


# def validate_parameter_constraints(parameter_constraints, params, caller_name):
#     """Validate types and values of given parameters.
#     Parameters
#     ----------
#     parameter_constraints : dict or {"no_validation"}
#         If "no_validation", validation is skipped for this parameter.
#         If a dict, it must be a dictionary `param_name: list of constraints`.
#         A parameter is valid if it satisfies one of the constraints from the list.
#         Constraints can be:
#         - an Interval object, representing a continuous or discrete range of numbers
#         - the string "array-like"
#         - the string "sparse matrix"
#         - the string "random_state"
#         - callable
#         - None, meaning that None is a valid value for the parameter
#         - any type, meaning that any instance of this type is valid
#         - an Options object, representing a set of elements of a given type
#         - a StrOptions object, representing a set of strings
#         - the string "boolean"
#         - the string "verbose"
#         - the string "cv_object"
#         - the string "missing_values"
#         - a HasMethods object, representing method(s) an object must have
#         - a Hidden object, representing a constraint not meant to be exposed to the user
#     params : dict
#         A dictionary `param_name: param_value`. The parameters to validate against the
#         constraints.
#     caller_name : str
#         The name of the estimator or function or method that called this function.
#     """
#     if len(set(parameter_constraints) - set(params)) != 0:
#         raise ValueError(
#             f"The parameter constraints {list(parameter_constraints)}"
#             " contain unexpected parameters"
#             f" {set(parameter_constraints) - set(params)}"
#         )

#     for param_name, param_val in params.items():
#         # We allow parameters to not have a constraint so that third party estimators
#         # can inherit from sklearn estimators without having to necessarily use the
#         # validation tools.
#         if param_name not in parameter_constraints:
#             continue

#         constraints = parameter_constraints[param_name]

#         if constraints == "no_validation":
#             continue

#         constraints = [make_constraint(constraint) for constraint in constraints]

#         for constraint in constraints:
#             if constraint.is_satisfied_by(param_val):
#                 # this constraint is satisfied, no need to check further.
#                 break
#         else:
#             # No constraint is satisfied, raise with an informative message.

#             # Ignore constraints that we don't want to expose in the error message,
#             # i.e. options that are for internal purpose or not officially supported.
#             constraints = [
#                 constraint for constraint in constraints if not constraint.hidden
#             ]

#             if len(constraints) == 1:
#                 constraints_str = f"{constraints[0]}"
#             else:
#                 constraints_str = (
#                     f"{', '.join([str(c) for c in constraints[:-1]])} or"
#                     f" {constraints[-1]}"
#                 )

#             raise ValueError(
#                 f"The {param_name!r} parameter of {caller_name} must be"
#                 f" {constraints_str}. Got {param_val!r} instead."
#             )


# #
# # class DiscreteIOBatchData(Dataset):
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
