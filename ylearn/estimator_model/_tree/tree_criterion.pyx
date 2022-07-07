# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

"""
By writing a criterion class inherited from the sklearn.tree._criterion, we can
implement the causal tree more directly.
"""
from libc.string cimport memcpy
from libc.string cimport memset
from libc.math cimport log
from libc.math cimport exp

import numpy as np
cimport numpy as np

np.import_array()

cdef double INFINITY = np.inf

from ylearn.sklearn_ex.cloned.tree._tree cimport DOUBLE_t
from ylearn.sklearn_ex.cloned.tree._tree cimport SIZE_t

from ylearn.sklearn_ex.cloned.tree._criterion cimport RegressionCriterion
from libc.stdio cimport printf
#-------------------------------------start of the implementation
cdef double eps = 1e-5
cdef double alpha = 1e10


cdef class HonestCMSE(RegressionCriterion):
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

        #-----------------        
        #
        self.weighted_n_node_samples = 0.0 # nn
        #
        self.nt_total = 0.0
        self.n0_total = 0.0

        #
        #self.weighted_n_left = 0.0 
        #
        self.nt_left = 0.0
        self.n0_left = 0.0
        
        #
        #self.weighted_n_right = 0.0
        #
        self.nt_right = 0.0
        self.n0_right = 0.0


        #
        #self.sq_sum_total = 0.0
        #
        self.yt_sq_sum_total = 0.0
        self.y0_sq_sum_total = 0.0

        #
        ## self.sum_total = np.zeros(n_outputs, dtype=np.float64)
        #
        self.yt_sum_total = np.zeros(n_outputs, dtype=np.float64)
        self.y0_sum_total = np.zeros(n_outputs, dtype=np.float64)

        #
        ## self.sum_left = np.zeros(n_outputs, dtype=np.float64)
        #
        self.yt_sum_left = np.zeros(n_outputs, dtype=np.float64)
        self.y0_sum_left = np.zeros(n_outputs, dtype=np.float64)

        #
        ## self.sum_right = np.zeros(n_outputs, dtype=np.float64)
        #
        self.yt_sum_right = np.zeros(n_outputs, dtype=np.float64)
        self.y0_sum_right = np.zeros(n_outputs, dtype=np.float64)
    
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
        #
        self.nt_total = 0.
        self.n0_total = 0.

        cdef SIZE_t i
        cdef SIZE_t p
        cdef DOUBLE_t y_ik
        #
        cdef DOUBLE_t wt_y_ik
        cdef DOUBLE_t w0_y_ik
        #
        cdef DOUBLE_t w

        #
        #self.sq_sum_total = 0.0
        #
        self.yt_sq_sum_total = 0.0
        self.y0_sq_sum_total = 0.0

        #
        #memset(&self.sum_total[0], 0, self.n_outputs * sizeof(double))
        #
        memset(&self.yt_sum_total[0], 0, self.n_outputs * sizeof(double))
        memset(&self.y0_sum_total[0], 0, self.n_outputs * sizeof(double))


        for p in range(start, end):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i] - eps

            y_ik = self.y[i, 0]

            #
            wt_y_ik = w * y_ik
            w0_y_ik = (1. - w) * y_ik

            #
            #self.sum_total[0] += wt_y_ik
            #
            self.yt_sum_total[0] += wt_y_ik
            self.y0_sum_total[0] += w0_y_ik

            #
            #self.sq_sum_total += wt_y_ik * y_ik
            #
            self.yt_sq_sum_total += wt_y_ik * y_ik
            self.y0_sq_sum_total += w0_y_ik * y_ik

            #
            self.weighted_n_node_samples += w
            #
            self.nt_total += w
            self.n0_total += (1. - w)

        # Reset to pos=start
        self.reset()
        return 0

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start."""
        cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
        #
        #memset(&self.sum_left[0], 0, n_bytes)
        #
        memset(&self.yt_sum_left[0], 0, n_bytes)
        memset(&self.y0_sum_left[0], 0, n_bytes)

        #
        #memcpy(&self.sum_right[0], &self.sum_total, n_bytes)
        #
        memcpy(&self.yt_sum_right[0], &self.yt_sum_total[0], n_bytes)
        memcpy(&self.y0_sum_right[0], &self.y0_sum_total[0], n_bytes)

        #
        #self.weighted_n_left = 0.0
        #
        self.nt_left = 0.0
        self.n0_left = 0.0

        #
        #self.weighted_n_right = self.weighted_n_node_samples
        #
        self.nt_right = self.nt_total
        self.n0_right = self.n0_total

        self.pos = self.start
        return 0

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criterion at pos=end."""
        cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
        #
        #memset(&self.sum_right[0], 0, n_bytes)
        #
        memset(&self.yt_sum_right[0], 0, n_bytes)
        memset(&self.y0_sum_right[0], 0, n_bytes)

        #
        #memcpy(&self.sum_left[0], &self.sum_total[0], n_bytes)
        #
        memcpy(&self.yt_sum_left[0], &self.yt_sum_total[0], n_bytes)
        memcpy(&self.y0_sum_left[0], &self.y0_sum_total[0], n_bytes)

        #
        #self.weighted_n_right = 0.0
        #
        self.nt_right = 0.0
        self.n0_right = 0.0

        #
        #self.weighted_n_left = self.weighted_n_node_samples
        #
        self.nt_left = self.nt_total
        self.n0_left = self.n0_total

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
        cdef DOUBLE_t w = 1.0

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

                #
                #self.sum_left[0] += w * self.y[i, 0]
                #
                self.yt_sum_left[0] += w * self.y[i, 0]
                self.y0_sum_left[0] += (1.0 - w) * self.y[i, 0]

                #
                #self.weighted_n_left += w
                #
                self.nt_left += w
                self.n0_left += (1.0 - w)
        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i] - eps

                #
                #self.sum_left[0] -= w * self.y[i, 0]
                #
                self.yt_sum_left[0] -= w * self.y[i, 0]
                self.y0_sum_left[0] -= (1.0 - w) * self.y[i, 0]

                #
                #self.weighted_n_left -= w
                #
                self.nt_left -= w
                self.n0_left -= (1.0 - w)

        #
        #self.weighted_n_right = (self.weighted_n_node_samples -
        #                         self.weighted_n_left)
        #
        self.nt_right = (self.nt_total - self.nt_left)
        self.n0_right = (self.n0_total - self.n0_left)

        #
        ##  self.sum_right[0] = self.sum_total - self.sum_left[0]
        #
        self.yt_sum_right[0] = self.yt_sum_total[0] - self.yt_sum_left[0]
        self.y0_sum_right[0] = self.y0_sum_total[0] - self.y0_sum_left[0]

        self.pos = new_pos
        return 0

    cdef double node_impurity(self) nogil:
        #printf("impurity\n")
        cdef double impurity
        #### FIXME
        # Note: the reason we initiate impurity as alpha is to prevent node_impurity
        # being negative, which whill then make the training of the tree stop immediately
        impurity = alpha
        impurity += 2 * (self.yt_sq_sum_total / (self.nt_total + eps) - (self.yt_sum_total[0] / (self.nt_total + eps))**2.0) / (self.nt_total + eps)
        impurity += 2 * (self.y0_sq_sum_total / (self.n0_total + eps) - (self.y0_sum_total[0] / (self.n0_total + eps))**2.0) / (self.n0_total + eps)
        impurity -= (self.yt_sum_total[0] / (self.nt_total + eps) - self.y0_sum_total[0] / (self.n0_total + eps))**2.0
        #printf("impurity value %f\n", impurity)
        return impurity

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t pos = self.pos
        cdef SIZE_t start = self.start

        cdef DOUBLE_t y_ik
        #
        #cdef double sq_sum_left = 0.0
        #
        cdef double yt_sq_sum_left = 0.0
        cdef double y0_sq_sum_left = 0.0

        #
        #cdef double sq_sum_right
        #
        cdef double yt_sq_sum_right
        cdef double y0_sq_sum_right

        cdef SIZE_t i
        cdef SIZE_t p
        cdef DOUBLE_t w = 1.0

        for p in range(start, pos):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i] - eps

            y_ik = self.y[i, 0]
            
            #
            #sq_sum_left += w * y_ik * y_ik
            #
            yt_sq_sum_left += w * y_ik * y_ik
            y0_sq_sum_left += (1. - w) * y_ik * y_ik

        #
        #sq_sum_right = self.sq_sum_total - sq_sum_left
        #
        yt_sq_sum_right = self.yt_sq_sum_total - yt_sq_sum_left
        y0_sq_sum_right = self.y0_sq_sum_total - y0_sq_sum_left

        #
        #impurity_left[0] = sq_sum_left / self.weighted_n_left
        #
        impurity_left[0] = alpha
        impurity_left[0] += 2 * (yt_sq_sum_left / (self.nt_left + eps) - (self.yt_sum_left[0] / (self.nt_left + eps))**2.0) / (self.nt_left + eps)
        impurity_left[0] += 2 * (y0_sq_sum_left / (self.n0_left + eps) - (self.y0_sum_left[0] / (self.n0_left + eps))**2.0) / (self.n0_left + eps)
        
        #
        #impurity_right[0] = sq_sum_right / self.weighted_n_right
        #
        impurity_right[0] = alpha
        impurity_right[0] += 2 * (yt_sq_sum_right / (self.nt_right + eps) - (self.yt_sum_right[0] / (self.nt_right + eps))**2.0) / (self.nt_right + eps)
        impurity_right[0] += 2 * (y0_sq_sum_right / (self.n0_right + eps) - (self.y0_sum_right[0] / (self.n0_right + eps))**2.0) / (self.n0_right + eps)
        
        #impurity_left[0] /= self.n_outputs
        #impurity_right[0] /= self.n_outputs
        impurity_left[0] -= (self.yt_sum_left[0] / (self.nt_left + eps) - self.y0_sum_left[0] / (self.n0_left + eps)) ** 2.0
        impurity_right[0] -= (self.yt_sum_right[0] / (self.nt_right + eps) - self.y0_sum_right[0] / (self.n0_right + eps)) ** 2.0 
    
    cdef void node_value(self, double* dest) nogil:
        #printf("node_value")
        """Compute the node value of samples[start:end] into dest."""
        #### FIXME
        dest[0] = self.yt_sum_total[0] / (self.nt_total + eps) - self.y0_sum_total[0] /  (self.n0_total + eps)
        #dest[0] = 0.0

    cdef double proxy_impurity_improvement(self) nogil:
        cdef double impurity_left
        cdef double impurity_right
        self.children_impurity(&impurity_left, &impurity_right)
        #printf("impurity left %f\n", impurity_left)
        #printf("impurity right %f\n", impurity_right)

        return (- (self.nt_right + self.n0_right) * impurity_right
                - (self.nt_left + self.n0_left) * impurity_left)

    cdef double impurity_improvement(self, double impurity_parent,
                                     double impurity_left,
                                     double impurity_right) nogil:
        #printf("improve\n")
        return (impurity_parent - ((self.nt_right + self.n0_right) / (self.nt_total + self.n0_total) * impurity_right)
                                - ((self.nt_left + self.n0_left) / (self.nt_total + self.n0_total) * impurity_left))


cdef class CMSE(HonestCMSE):
    cdef double node_impurity(self) nogil:
        #printf("impurity\n")
        cdef double impurity
        #### FIXME
        #impurity = log(1 + exp(- (self.yt_sum_total[0] / (self.nt_total + eps) - self.y0_sum_total[0] / (self.n0_total + eps))**2.0))
        #printf("impurity value %f\n", impurity)
        #impurity = (self.yt_sq_sum_total + self.y0_sq_sum_total) / (self.nt_total + self.n0_total)
        impurity = alpha
        impurity -= (self.yt_sum_total[0] / (self.nt_total + eps) - self.y0_sum_total[0] / (self.n0_total + eps))**2.0
        return impurity

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        #impurity_left[0] = log(1 + exp(- (self.yt_sum_left[0] / (self.nt_left + eps) - self.y0_sum_left[0] / (self.n0_left + eps)) ** 2.0))
        #impurity_right[0] = log(1 + exp( - (self.yt_sum_right[0] / (self.nt_right + eps) - self.y0_sum_right[0] / (self.n0_right + eps)) ** 2.0))
        
        impurity_left[0] = alpha
        impurity_right[0] = alpha
        impurity_left[0] -= (self.yt_sum_left[0] / (self.nt_left + eps) - self.y0_sum_left[0] / (self.n0_left + eps)) ** 2.0
        impurity_right[0] -= (self.yt_sum_right[0] / (self.nt_right + eps) - self.y0_sum_right[0] / (self.n0_right + eps)) ** 2.0

    cdef void node_value(self, double* dest) nogil:
        #printf("node_value")
        """Compute the node value of samples[start:end] into dest."""
        #### FIXME
        dest[0] = self.yt_sum_total[0] / (self.nt_total + eps) - self.y0_sum_total[0] /  (self.n0_total + eps)
        #dest[0] = 0.0

    cdef double proxy_impurity_improvement(self) nogil:
        cdef double impurity_left
        cdef double impurity_right
        self.children_impurity(&impurity_left, &impurity_right)
        #printf("impurity left %f\n", impurity_left)
        #printf("impurity right %f\n", impurity_right)

        return (- (self.nt_right + self.n0_right) * impurity_right
                - (self.nt_left + self.n0_left) * impurity_left)

    cdef double impurity_improvement(self, double impurity_parent,
                                     double impurity_left,
                                     double impurity_right) nogil:
        #printf("improve\n")
        return (impurity_parent - ((self.nt_right + self.n0_right) / (self.nt_total + self.n0_total) * impurity_right)
                                - ((self.nt_left + self.n0_left) / (self.nt_total + self.n0_total) * impurity_left))


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
        printf("node impurity\n")
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
        printf("children impurity\n")
        for p in range(start, pos):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                y_ik = self.y[i, k]
                sq_sum_left += w * y_ik * y_ik

        sq_sum_right = self.sq_sum_total - sq_sum_left

        impurity_left[0] = sq_sum_left / self.weighted_n_left
        impurity_right[0] = sq_sum_right / self.weighted_n_right

        for k in range(self.n_outputs):
            impurity_left[0] -= (self.sum_left[k] / self.weighted_n_left) ** 2.0
            impurity_right[0] -= (self.sum_right[k] / self.weighted_n_right) ** 2.0

        impurity_left[0] /= self.n_outputs
        impurity_right[0] /= self.n_outputs