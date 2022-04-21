"""
By writing a criterion class inherited from the sklearn.tree._criterion, we can
implement the causal tree more directly.
"""

from sklearn.tree._criterion cimport RegressionCriterion
from sklearn.tree._criterion cimport SIZE_t, DOUBLE_t

// TODO: note the zerodivision error
cdef class CMSE(RegressionCriterion):
    r"""Mean squared error impurity criterion for the honest approach of
    estimating causal effects using CART[1].
        CMSE = left_ce + right_ce,
    where (the left and right ce are similar), by defining
    TODO: consider the coefficient (1 / n + 1 / n_est)
    r = n_t / (n_t + n_0),
        left_ce = tau^2 - (1 / n_t + 1 / n_0) * (var_t / r + var_0 / (1 - r)) 
    and 
        tau = \sum_i yt_i / n_t - \sum_j y0_j / n_0,
        var_t = (\sum_i yt^2_i) / n_t - (\sum_i yt_i / n_t)^2,
        var_0 = (\sum_j y0^2_j) / n_0 - (\sum_j y0_j / n_0)^2.

    Reference
    ----------
    [1] Recursive partitioning for heterogeneous causal effects. 
        S. Athey & G. Imbens.
    """

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node.
        Evaluate the CMSE criterion as impurity of the current node,
        i.e. the impurity of samples[start:end]. The smaller the impurity the
        better.
        """
        cdef double impurity

        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end
        cdef SIZE_t i
        cdef SIZE_t p
        cdef DOUBLE_t y_ik
        cdef DOUBLE_t w

        cdef SIZE_t* samples = self.samples
        cdef DOUBLE_t* sample_weight = self.sample_weight

        cdef double n_t = 0.0  # number of treatment examples
        cdef double n_0 = 0.0  # number of control examples
        cdef double yt_sum = 0.0  # sum of yt
        cdef double y0_sum = 0.0  # sum of y0
        cdef double yt_sq_sum = 0.0  # sum of yt^2
        cdef double y0_sq_sum = 0.0  # sum of y0^2
        cdef double mu_t  # estimation of yt
        cdef double mu_0  # estimation of y0
        cdef double var_t  # variance of the treatment group
        cdef double var_0  # variance of the control group
        cdef double tau

        for p in range(start, end):
            i = samples[p]
            w = sample_weight[i]

            n_t += w
            n_0 += (1 - w)
            # TODO: multi-output outcome
            y_ik = self.y[i, 0]
            yt_sum += y_ik * w
            yt_sq_sum += y_ik * y_ik * w
            y0_sum += y_ik * (1 - w)
            y0_sq_sum += y_ik * y_ik * (1 - w)
        mu_t = yt_sum / (n_t + 1e-5)
        mu_0 = y0_sum / (n_0 + 1e-5)
        tau = mu_t - mu_0
        var_t = yt_sq_sum / (n_t + 1e-5) - mu_t * mu_t
        var_0 = y0_sq_sum / (n_0 + 1e-5) - mu_0 * mu_0
        impurity = -tau * tau + (var_t / (n_t + 1e-5) + var_0 / (n_0 + 1e-5))
        return impurity

    cdef void children_impurity(self, double * impurity_left,
                                double * impurity_right) nogil:
        """Evaluate the impurity in children nodes.
        i.e. the impurity of the left child (samples[start:pos]) and the
        impurity the right child (samples[pos:end]).
        """
        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t pos = self.pos
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end
        cdef DOUBLE_t y_ik

        cdef double n_t_left = 0.0  # number of treatment examples
        cdef double n_0_left = 0.0  # number of control examples
        cdef double yt_sum_left = 0.0  # sum of yt
        cdef double y0_sum_left = 0.0  # sum of y0
        cdef double yt_sq_sum_left = 0.0  # sum of yt^2
        cdef double y0_sq_sum_left = 0.0  # sum of y0^2
        cdef double mu_t_left  # estimation of yt
        cdef double mu_0_left  # estimation of y0
        cdef double var_t_left  # variance of the treatment group
        cdef double var_0_left  # variance of the control group
        cdef double tau_left

        cdef double n_t_right = 0.0  # number of treatment examples
        cdef double n_0_right = 0.0  # number of control examples
        cdef double yt_sum_right = 0.0  # sum of yt
        cdef double y0_sum_right = 0.0  # sum of y0
        cdef double yt_sq_sum_right = 0.0  # sum of yt^2
        cdef double y0_sq_sum_right = 0.0  # sum of y0^2
        cdef double mu_t_right  # estimation of yt
        cdef double mu_0_right  # estimation of y0
        cdef double var_t_right  # variance of the treatment group
        cdef double var_0_right  # variance of the control group
        cdef double tau_right

        cdef SIZE_t i
        cdef SIZE_t p
        cdef DOUBLE_t w = 1.0

        for p in range(start, pos):
            i = samples[p]
            w = sample_weight[i]

            n_t_left += w
            n_0_left += (1 - w)
            y_ik = self.y[i, 0]
            yt_sum_left += y_ik * w
            yt_sq_sum_left += y_ik * y_ik * w
            y0_sum_left += y_ik * (1 - w)
            y0_sq_sum_left += y_ik * y_ik * (1 - w)

        for p in range(pos, end):
            i = samples[p]
            w = sample_weight[i]

            n_t_right += w
            n_0_right += (1 - w)
            y_ik = self.y[i, 0]
            yt_sum_right += y_ik * w
            yt_sq_sum_right += y_ik * y_ik * w
            y0_sum_right += y_ik * (1 - w)
            y0_sq_sum_right += y_ik * y_ik * (1 - w)

        mu_t_left = yt_sum_left / n_t_left
        mu_0_left = y0_sum_left / n_0_left
        tau_left = mu_t_left - mu_0_left
        var_t_left = yt_sq_sum_left / n_t_left - mu_t_left * mu_t_left
        var_0_left = y0_sq_sum_left / n_0_left - mu_0_left * mu_0_left
        impurity_left[0] = -tau_left * tau_left + (var_t_left / n_t_left + var_0_left / n_0_left)

        mu_t_right = yt_sum_right / n_t_right
        mu_0_right = y0_sum_right / n_0_right
        tau_right = mu_t_right - mu_0_right
        var_t_right = yt_sq_sum_right / n_t_right - mu_t_right * mu_t_right
        var_0_right = y0_sq_sum_right / n_0_right - mu_0_right * mu_0_right
        impurity_right[0] = -tau_right * tau_right + (var_t_right / n_t_right + var_0_right / n_0_right)

    cdef void node_value(self, double * dest) nogil:
        """Compute the node value of samples[start:end] into dest."""
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end
        cdef SIZE_t i
        cdef SIZE_t p
        cdef DOUBLE_t y_ik
        cdef DOUBLE_t w

        cdef SIZE_t* samples = self.samples
        cdef DOUBLE_t* sample_weight = self.sample_weight

        cdef double n_t = 0.0  # number of treatment examples
        cdef double n_0 = 0.0  # number of control examples
        cdef double yt_sum = 0.0  # sum of yt
        cdef double y0_sum = 0.0  # sum of y0

        for p in range(start, end):
            i = samples[p]
            w = sample_weight[i]

            n_t += w
            n_0 += (1 - w)
            y_ik = self.y[i, 0]
            yt_sum += y_ik * w
            y0_sum += y_ik * (1 - w)

        dest[0] = yt_sum / (n_t + 1e-5) - y0_sum / (n_0 + 1e-5)
