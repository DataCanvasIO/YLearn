from weakref import proxy
import numpy as np

from numpy.linalg import lstsq
from sklearn.utils import check_random_state

from .utils import grad, grad_coef, inverse_grad

INF = np.inf
MINF = -INF


class Node:
    def __init__(self, value=None, left=None, right=None, split=None):
        self.value = value
        self.left = left
        self.right = right
        self.feature = split[0]
        self.threshold = split[1]

    @property
    def _is_leaf(self):
        return True if (self.left is not None and self.right is not None) else False


class Leaf:
    def __init__(self, value):
        """
        `value` is an array of class probabilities if classifier is True, else
        the mean of the region
        """
        self.value = value


class _GrfTree:
    def __init__(
        self,
        *,
        max_depth=None,
        random_state=2022,
        min_split_tolerance=1e-10,
    ):
        """
        Parameters
        ----------
        max_depth: int or None
            The depth at which to stop growing the tree. If None, grow the tree
            until all leaves are pure. Default is None.
        n_feats : int
            Specifies the number of features to sample on each split. If None,
            use all features on each split. Default is None.
        criterion : {'mse', 'entropy', 'gini'}
            The error criterion to use when calculating splits. When
            `classifier` is False, valid entries are {'mse'}. When `classifier`
            is True, valid entries are {'entropy', 'gini'}. Default is
            'entropy'.
        random_state : int
        """
        self.random_state = random_state

        self.depth = 0
        self.root = None
        self.min_split_tolerance = min_split_tolerance

        self.max_depth = max_depth if max_depth else INF

    def check_data(self, data, outcome, treatment, adjustment, covariate):
        """Return transformed data in the form of array.

        Parameters
        ----------
        data : pd.DataFrame
            _description_
        outcome : str or list of str
            Names of outcome
        treatment : str or list of str
            Names of treatment
        adjustment : str or list of str
            Names of adjustment set, by default None
        covariate : str or list of str
            Names of covariat set, by default None
        """
        pass

    def fit(
        self,
        data,
        outcome,
        treatment,
        adjustment=None,
        covariate=None,
    ):
        # Note that this check_data should return arrays with at least 2 dimensions
        x, y, w, v = self.check_data(data, outcome, treatment, adjustment, covariate)

        if y.ndim > 1:
            assert y.shape[1] == 1, f"Currently only support scalar outcome"
        y = y.squeeze()

        # TODO: may consider add intercept to treatment matrix x

        self._fit_with_array(x, y, w, v)

        return self

    def predict(self, data=None):
        """
        Use the trained decision tree to classify or predict the examples in `X`.
        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            The training data of `N` examples, each with `M` features
        Returns
        -------
        preds : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
            The integer class labels predicted for each example in `X` if
            self.classifier = True, otherwise the predicted target values.
        """
        assert self._is_fitted, "The model is not fitted yet"
        w, v = self.check_data(data, self.covariate)
        return self._predict_with_array(w, v)

    def _fit_with_array(self, x, y, w, v):
        """
        Parameters
        ----------
        x : :py:class:`ndarray` of shape `(n, p)`
            The treatment vector of the training data of `n` examples, each with `p` dimensions, e.g., p-dimension one-hot vector if discrete treatment is assumed
        y : :py:class:`ndarray` of shape `(n,)`
            The outcome vector of the training data of `n` examples
        w : :py:class:`ndarray` of shape `(n, p)`
            The adjustment vector aka confounders of the training data
        v : :py:class:`ndarray` of shape `(n, p)`
            The covariate vector of the training data specifying hetergeneity
        """
        self.root = self._build_tree(x, y, w, v)
        self._is_fitted = True

    def _predict_with_array(self, w, v):
        return np.array([self._traverse(v_i, self.root) for v_i in v])

    def _build_tree(self, x, y, w, v, cur_depth=0):
        # return a leaf if has only one sample
        if len(y) == 1:
            return Node(value=y)

        # return a leaf if have reached max_depth
        if cur_depth >= self.max_depth:
            value = y.mean()
            return Node(value=value)

        n, v_d = v.shape

        # find the coef of the least square regression of y on x
        ls_coef = lstsq(x, y.squeeze())[0]

        # return a leaf if further splitting is meaningless
        x_dif = x - x.mean(axis=0)
        y_dif = y - y.mean()
        rho_ = grad_coef(x_dif, y_dif, ls_coef)
        if np.abs(rho_ - rho_[0]).max() >= self.min_split_tolerance:
            return Node(value=y[0])

        cur_depth += 1
        self.depth = max(self.depth, cur_depth)

        # label the tree value for further splitting
        rho = self._label_node(x_dif=x_dif, n=n, rho_=rho_)

        # Now we run the CART regression on rho
        # find the greedy split rule
        split_idx, thresh = self._split(v, rho)
        l = np.argwhere(v[:, split_idx] <= thresh).squeeze()
        r = np.argwhere(v[:, split_idx] > thresh).squeeze()

        # grow the children based on the split
        left_child = self._build_tree(
            x[l, :], y[l, :], w[l, :], v[l, :], cur_depth=cur_depth
        )
        right_child = self._build_tree(
            x[r, :], y[r, :], w[r, :], v[r, :], cur_depth=cur_depth
        )
        return Node(left=left_child, right=right_child, split=(split_idx, thresh))

    def _label_node(self, x_dif, n, rho_):
        grad_ = grad(x_dif=x_dif, n=n)
        inv_grad = inverse_grad(grad_)
        return np.einsum("ij,nj->ni", inv_grad, rho_).sum(1)

    def _split(self, v, rho):
        """Find the optimal split rule

        Parameters
        ----------
        v : np.ndarray
        rho : np.ndarray

        Returns
        -------
        _type_
            _description_
        """
        best_improve = MINF
        split_idx, split_thre = None, None
        sum_total = rho.sum()

        for i, v_i in enumerate(v.T):
            levels = np.unique(v_i)

            # build thresholds
            thre = (levels[:-1] + levels[1:]) / 2 if len(levels) > 1 else levels

            # find valid thresholds
            improve = np.array(
                [self._proxy_impurity_improvement(sum_total, rho, t, v_i) for t in thre]
            )
            max_improve, arg_max_improve = improve.max(), improve.argmax()
            if max_improve > best_improve:
                split_idx = i
                best_improve = max_improve
                split_thre = thre[arg_max_improve]

        return split_idx, split_thre

    def _proxy_impurity_improvement(self, sum_total, rho, threshold, split_feature):
        left_child = rho[np.argwhere(split_feature <= threshold).squeeze()]
        sum_left = left_child.sum()
        sum_right = sum_total - sum_left
        n_left = len(left_child)
        n_right = len(rho) - n_left
        return sum_left**2 / n_left + sum_right**2 / n_right

    def _traverse(self, X, node):
        if node._is_leaf:
            return node.value
        if X[node.feature] <= node.threshold:
            return self._traverse(X, node.left)
        return self._traverse(X, node.right)
