"""
By writing a criterion class inherited from the sklearn.tree._criterion, we can
implement the causal tree more directly.
"""
from libc.string cimport memcpy
from libc.string cimport memset

import numpy as np
cimport numpy as np
np.import_array()

from sklearn.tree._criterion cimport RegressionCriterion, Criterion
from sklearn.tree._criterion cimport SIZE_t, DOUBLE_t
from libc.stdio cimport printf


cdef double eps = 1e-6


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

            w = sample_weight[i] - 1

            n_t += w
            n_0 += (1. - w)
            y_ik = self.y[i, 0]
            yt_sum += y_ik * w
            y0_sum += y_ik * (1. - w)
        
        printf("yt %f\n", yt_sum)
        printf("y0 %f\n", y0_sum)
        printf("sum %f\n", self.sum_total)
        dest[0] = yt_sum / n_t - y0_sum / n_0

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
            
            w = sample_weight[i] - 1

            n_t += w
            n_0 += (1. - w)
            # TODO: multi-output outcome
            y_ik = self.y[i, 0]
            yt_sum += y_ik * w
            yt_sq_sum += y_ik * y_ik * w
            y0_sum += y_ik * (1. - w)
            y0_sq_sum += y_ik * y_ik * (1. - w)

        printf("yt %f\n", yt_sum)
        printf("y0 %f\n", y0_sum)
        printf("sum %f\n", self.sum_total)
        printf("diff %f\n", self.sum_total)
        mu_t = yt_sum / n_t
        mu_0 = y0_sum / n_0
        tau = mu_t - mu_0
        var_t = yt_sq_sum / n_t - mu_t * mu_t
        var_0 = y0_sq_sum / n_0 - mu_0 * mu_0
        impurity = -tau * tau + (var_t / n_t + var_0 / n_0)
        return impurity

    cdef void children_impurity(self, double * impurity_left, double * impurity_right) nogil:
        """Evaluate the impurity in children nodes.
        i.e. the impurity of the left child (samples[start:pos]) and the
        impurity the right child (samples[pos:end]).
        """
        printf("entering children")
        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t pos = self.pos
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end
        cdef DOUBLE_t y_ik
        cdef DOUBLE_t w

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

        for p in range(start, end):
            i = samples[p]
            w = sample_weight[i] - 1
            y_ik = self.y[i, 0]

            n_t_left += w
            n_0_left += (1. - w)
            y_ik = self.y[i, 0]
            yt_sum_left += y_ik * w
            yt_sq_sum_left += y_ik * y_ik * w
            y0_sum_left += y_ik * (1. - w)
            y0_sq_sum_left += y_ik * y_ik * (1. - w)

        for p in range(pos, end):
            i = samples[p]
            w = sample_weight[i] - 1

            n_t_right += w
            n_0_right += (1. - w)
            y_ik = self.y[i, 0]
            yt_sum_right += y_ik * w
            yt_sq_sum_right += y_ik * y_ik * w
            y0_sum_right += y_ik * (1. - w)
            y0_sq_sum_right += y_ik * y_ik * (1. - w)

        mu_t_left = yt_sum_left / n_t_left
        mu_0_left = y0_sum_left / n_0_left
        tau_left = mu_t_left - mu_0_left
        var_t_left = yt_sq_sum_left / n_t_left - mu_t_left * mu_t_left
        var_0_left = y0_sq_sum_left / n_0_left - mu_0_left * mu_0_left
        impurity_left[0] = -tau_left * tau_left + (var_t_left / n_t_left + var_0_left / n_0_left)
        printf("impurity left %f\n", impurity_left[0])


        mu_t_right = yt_sum_right / n_t_right
        mu_0_right = y0_sum_right / n_0_right
        tau_right = mu_t_right - mu_0_right
        var_t_right = yt_sq_sum_right / n_t_right - mu_t_right * mu_t_right
        var_0_right = y0_sq_sum_right / n_0_right - mu_0_right * mu_0_right
        impurity_right[0] = -tau_right * tau_right + (var_t_right / n_t_right + var_0_right / n_0_right)
        printf("impurity right %f\n", impurity_right[0])


cdef class MSE(RegressionCriterion):
    """Mean squared error impurity criterion.
        MSE = var_left + var_right
    """

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node.
        Evaluate the MSE criterion as impurity of the current node,
        i.e. the impurity of samples[start:end]. The smaller the impurity the
        better.
        """
        cdef double impurity
        cdef SIZE_t k

        impurity = self.sq_sum_total / self.weighted_n_node_samples
        for k in range(self.n_outputs):
            impurity -= (self.sum_total[k] / self.weighted_n_node_samples)**2.0

        return impurity / self.n_outputs

    cdef double proxy_impurity_improvement(self) nogil:
        """Compute a proxy of the impurity reduction.
        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.
        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        The MSE proxy is derived from
            sum_{i left}(y_i - y_pred_L)^2 + sum_{i right}(y_i - y_pred_R)^2
            = sum(y_i^2) - n_L * mean_{i left}(y_i)^2 - n_R * mean_{i right}(y_i)^2
        Neglecting constant terms, this gives:
            - 1/n_L * sum_{i left}(y_i)^2 - 1/n_R * sum_{i right}(y_i)^2
        """
        cdef SIZE_t k
        cdef double proxy_impurity_left = 0.0
        cdef double proxy_impurity_right = 0.0

        for k in range(self.n_outputs):
            proxy_impurity_left += self.sum_left[k] * self.sum_left[k]
            proxy_impurity_right += self.sum_right[k] * self.sum_right[k]

        return (proxy_impurity_left / self.weighted_n_left +
                proxy_impurity_right / self.weighted_n_right)

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes.
        i.e. the impurity of the left child (samples[start:pos]) and the
        impurity the right child (samples[pos:end]).
        """
        printf("entering children\n")
        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t pos = self.pos
        cdef SIZE_t start = self.start

        cdef DOUBLE_t y_ik

        cdef double sq_sum_left = 0.0
        cdef double sq_sum_right

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t w = 1.0

        for p in range(start, pos):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i] - 1

            for k in range(self.n_outputs):
                y_ik = self.y[i, k]
                sq_sum_left += w * y_ik * y_ik

        sq_sum_right = self.sq_sum_total - sq_sum_left

        impurity_left[0] = sq_sum_left / self.weighted_n_left
        impurity_right[0] = sq_sum_right / self.weighted_n_right

        for k in range(self.n_outputs):
            impurity_left[0] -= (self.sum_left[k] / self.weighted_n_left) ** 2.0
            impurity_right[0] -= (self.sum_right[k] / self.weighted_n_right) ** 2.0
        
        printf("sum_total %f\n", self.sum_total)

        impurity_left[0] /= self.n_outputs
        impurity_right[0] /= self.n_outputs


#-------------------------------------start of a new implementation

cdef class CMSE_New(Criterion):
    def __cinit__(self, SIZE_t n_outputs, SIZE_t n_samples):
        """Initialize parameters for this criterion.
        Parameters
        ----------
        n_outputs : SIZE_t
            The number of targets to be predicted
        n_samples : SIZE_t
            The total number of samples to fit on
        """
        # Default values
        self.sample_weight = NULL

        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0 # nn
        #
        self.n_t_total = 0.0
        self.n_0_total = 0.0

        self.weighted_n_left = 0.0 
        self.weighted_n_right = 0.0
        #
        self.n_t_left = 0.0
        self.n_0_left = 0.0
        self.n_t_right = 0.0
        self.n_0_right = 0.0


        #
        self.sq_sum_total = 0.0
        self.yt_sq_sum_total = 0.0
        self.y0_sq_sum_total = 0.0

        #
        #self.sum_total = np.zeros(n_outputs, dtype=np.float64)
        #self.yt_sum_total = np.zeros(n_outputs, dtype=np.float64)
        #self.y0_sum_total = np.zeros(n_outputs, dtype=np.float64)
        self.sum_total = 0.0
        self.yt_sum_total = 0.0
        self.y0_sum_totaol = 0.0
        #
        #self.sum_left = np.zeros(n_outputs, dtype=np.float64)
        #self.yt_sum_left = np.zeros(n_outputs, dtype=np.float64)
        #self.y0_sum_left = np.zeros(n_outputs, dtype=np.float64)
        self.sum_left = 0.0
        self.yt_sum_left = 0.0
        self.y0_sum_left = 0.0

        #
        #self.sum_right = np.zeros(n_outputs, dtype=np.float64)
        #self.yt_sum_right = np.zeros(n_outputs, dtype=np.float64)
        #self.y0_sum_right = np.zeros(n_outputs, dtype=np.float64)
        self.sum_right = 0.0
        self.yt_sum_right = 0.0
        self.y0_sum_right = 0.0
    
    def __reduce__(self):
        return (type(self), (self.n_outputs, self.n_samples), self.__getstate__())
    
    cdef int init(self, const DOUBLE_t[:, ::1] y, DOUBLE_t* sample_weight,
                  double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                  SIZE_t end) nogil except -1:
        """Initialize the criterion.
        This initializes the criterion at node samples[start:end] and children
        samples[start:start] and samples[start:end].
        """
        # Initialize fields
        self.y = y
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.
        self.n_t_total = 0.0
        self.n_0_total = 0.0

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t y_ik
        cdef DOUBLE_t w_y_ik
        cdef DOUBLE_t w
        cdef double eps = 1e-6
        self.sq_sum_total = 0.0
        self.yt_sq_sum_total = 0.0
        self.y0_sq_sum_total = 0.0

        memset(&self.sum_total, 0, self.n_outputs * sizeof(double))
        memset(&self.yt_sum_total, 0, self.n_outputs * sizeof(double))
        memset(&self.y0_sum_total, 0, self.n_outputs * sizeof(double))


        for p in range(start, end):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i] - eps

            y_ik = self.y[i, 0]
            #
            wt_y_ik = w * y_ik
            w0_y_ik = (1.0 - w) * y_ik
            self.sum_total += wt_y_ik
            self.sq_sum_total += wt_y_ik * y_ik
            #
            self.yt_sum_total += wt_y_ik
            self.y0_sum_total += w0_y_ik
            self.yt_sq_sum_total += wt_y_ik * y_ik
            self.y0_sq_sum_total += w0_y_ik * y_ik

            self.weighted_n_node_samples += w
            self.n_t_total += w
            self.n_0_total += (1.0 - w)

        # Reset to pos=start
        self.reset()
        return 0

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start."""
        cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
        memset(&self.sum_left[0], 0, n_bytes)
        memcpy(&self.sum_right[0], &self.sum_total, n_bytes)
        #
        memset(&self.yt_sum_left[0], 0, n_bytes)
        memcpy(&self.yt_sum_right[0], &self.yt_sum_total, n_bytes)
        
        memset(&self.y0_sum_left[0], 0, n_bytes)
        memcpy(&self.y0_sum_right[0], &self.y0_sum_total, n_bytes)

        self.weighted_n_left = 0.0
        #
        self.n_t_left = 0.0
        self.n_0_left = 0.0

        self.weighted_n_right = self.weighted_n_node_samples
        #
        self.n_t_right = self.n_t_total
        self.n_0_right = self.n_0_total

        self.pos = self.start
        return 0

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criterion at pos=end."""
        cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
        memset(&self.sum_right[0], 0, n_bytes)
        memcpy(&self.sum_left[0], &self.sum_total, n_bytes)
        #
        memset(&self.yt_sum_right[0], 0, n_bytes)
        memcpy(&self.yt_sum_left[0], &self.yt_sum_total, n_bytes)
        
        memset(&self.y0_sum_right[0], 0, n_bytes)
        memcpy(&self.y0_sum_left[0], &self.y0_sum_total, n_bytes)

        self.weighted_n_right = 0.0
        #
        self.n_t_right = 0.0
        self.n_0_right = 0.0

        self.weighted_n_left = self.weighted_n_node_samples
        #
        self.n_t_left = self.n_t_total
        self.n_0_left = self.n_0_total

        self.pos = self.end
        return 0

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left."""
        cdef double* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples

        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end
        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t w
        cdef double eps = 1e-6

        # Update statistics up to new_pos
        #
        # Given that
        #           sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_pos.
        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i] - eps

                self.sum_left[0] += w * self.y[i, 0]
                #
                self.yt_sum_left[0] += w * self.y[i, 0]
                self.y0_sum_left[0] += (1.0 - w) * self.y[i, 0]

                self.weighted_n_left += w
                #
                self.n_t_left += w
                self.n_0_left += (1.0 - w)
        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i] - eps

                self.sum_left[0] -= w * self.y[i, 0]
                #
                self.yt_sum_left[0] -= w * self.y[i, 0]
                self.y0_sum_left[0] -= (1.0 - w) * self.y[i, 0]

                self.weighted_n_left -= w
                #
                self.n_t_left -= w
                self.n_0_left -= (1.0 - w)

        self.weighted_n_right = (self.weighted_n_node_samples -
                                 self.weighted_n_left)
        #
        self.n_t_right = (self.n_t_total - self.n_t_left)
        self.n_0_right = (self.n_0_total - self.n_0_left)

        self.sum_right[0] = self.sum_total - self.sum_left[0]
        #
        self.yt_sum_right[0] = self.yt_sum_total - self.yt_sum_left[0]
        self.y0_sum_right[0] = self.y0_sum_total - self.y0_sum_left[0]

        self.pos = new_pos
        return 0

    cdef double node_impurity(self) nogil:
        cdef double eps = 1e-6
        cdef double impurity
        return ((self.yt_sq_sum_total / (self.n_t_total + eps) - (self.yt_sum_total / (self.n_t_total + eps))**2.0) / (self.n_t_total + eps)
            + (self.y0_sq_sum_total / (self.n_0_total + eps) - (self.y0_sum_total / (self.n_0_total + eps))**2.0) / (self.n_0_total + eps)
            - (self.yt_sum_total / (self.n_t_total + eps) - self.y0_sum_total / (self.n_0_total + eps))**2.0)

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t pos = self.pos
        cdef SIZE_t start = self.start

        cdef DOUBLE_t y_ik

        cdef double sq_sum_left = 0.0
        cdef double sq_sum_right
        #
        cdef double eps = 1e-6
        cdef double yt_sq_sum_left = 0.0
        cdef double y0_sq_sum_left = 0.0
        cdef double yt_sq_sum_right
        cdef double y0_sq_sum_right

        cdef SIZE_t i
        cdef SIZE_t p
        cdef DOUBLE_t w = 1.0

        for p in range(start, pos):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i] - eps

            y_ik = self.y[i, k]
            sq_sum_left += w * y_ik * y_ik
            #
            yt_sq_sum_left += w * y_ik * y_ik
            y0_sq_sum_left += (1. - w) * yik * yik

        sq_sum_right = self.sq_sum_total - sq_sum_left
        #
        yt_sq_sum_right = self.yt_sq_sum_total - yt_sq_sum_left
        y0_sq_sum_right = self.y0_sq_sum_total - y0_sq_sum_right

        #impurity_left[0] = sq_sum_left / self.weighted_n_left
        #impurity_right[0] = sq_sum_right / self.weighted_n_right
        
        impurity_left[0] = (yt_sq_sum_left / (self.n_t_left + eps) - (self.yt_sum_left[0] / (self.n_t_left + eps))**2.0) / self.n_t_left
        impurity_left[0] += (y0_sq_sum_left / (self.n_0_left + eps) - (self.y0_sum_left[0] / (self.n_0_left + eps))**2.0) / self.n_0_left
        impurity_right[0] = (yt_sq_sum_right / (self.n_t_right + eps) - (self.yt_sum_right[0] / (self.n_t_right + eps))**2.0) / self.n_t_right
        impurity_right[0] += (y0_sq_sum_right / (self.n_0_right + eps) - (self.y0_sum_right[0] / (self.n_0_right + eps))**2.0) / self.n_0_right
        #impurity_left[0] -= (self.sum_left[0] / self.weighted_n_left) ** 2.0

        #impurity_right[0] -= (self.sum_right[0] / self.weighted_n_right) ** 2.0

        #impurity_left[0] /= self.n_outputs
        #impurity_right[0] /= self.n_outputs
        impurity_left[0] -= (self.yt_sum_left[0] / (self.n_t_left + eps) - self.y0_sum_left[0] / (self.n_0_left + eps)) ** 2.0
        impurity_right[0] -= (self.yt_sum_right[0] / (self.n_t_right + eps) - self.y0_sum_right[0] / (self.n_0_right + eps)) ** 2.0
    
    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] into dest."""
        cdef double eps = 1e-6
        #dest[0] = self.sum_total / self.weighted_n_node_samples
        #
        dest[0] = self.yt_sum_total / (self.n_t_total + eps) - self.y0_sum_total /  (self.n_0_total + eps)

    cdef double proxy_impurity_improvement(self) nogil:
        """Compute a proxy of the impurity reduction.
        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.
        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        The MSE proxy is derived from
            sum_{i left}(y_i - y_pred_L)^2 + sum_{i right}(y_i - y_pred_R)^2
            = sum(y_i^2) - n_L * mean_{i left}(y_i)^2 - n_R * mean_{i right}(y_i)^2
        Neglecting constant terms, this gives:
            - 1/n_L * sum_{i left}(y_i)^2 - 1/n_R * sum_{i right}(y_i)^2
        """
        #
        cdef double impurity_left
        cdef double impurity_right

        cdef double n_total = 0.0
        cdef double n_right = 0.0
        cdef double n_left = 0.0

        n_left += (self.n_t_left + self.n_0_left)
        n_right += (self.n_t_right + self.n_0_right)
        n_total += (n_left + n_right)

        self.children_impurity(&impurity_left, &impurity_right)

        return (- n_right / n_total * impurity_right
                - n_left / n_total * impurity_left)

    cdef double impurity_improvement(self, double impurity_parent,
                                     double impurity_left,
                                     double impurity_right) nogil:
        cdef double n_total = 0.0
        cdef double n_right = 0.0
        cdef double n_left = 0.0

        n_left += (self.n_t_left + self.n_0_left)
        n_right += (self.n_t_right + self.n_0_right)
        n_total += (n_left + n_right)
        return (impurity_parent - n_right / n_total * impurity_right
                - self.n_left / n_total * impurity_left)