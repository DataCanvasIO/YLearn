"""
By writing a criterion class inherited from the sklearn.tree._criterion, we can
implement the causal tree more directly.
"""
from libc.string cimport memcpy
from libc.string cimport memset

import numpy as np
cimport numpy as np

np.import_array()

from sklearn.tree._criterion cimport RegressionCriterion
from libc.stdio cimport printf
#-------------------------------------start of a new implementation
cdef double eps = 1e-6

cdef class CMSE(RegressionCriterion):
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
        self.weighted_n_left = 0.0 
        #
        self.nt_left = 0.0
        self.n0_left = 0.0
        
        #
        self.weighted_n_right = 0.0
        #
        self.nt_right = 0.0
        self.n0_right = 0.0


        #
        self.sq_sum_total = 0.0
        #
        self.yt_sq_sum_total = 0.0
        self.y0_sq_sum_total = 0.0

        #
        self.sum_total = np.zeros(n_outputs, dtype=np.float64)
        #
        self.yt_sum_total = np.zeros(n_outputs, dtype=np.float64)
        self.y0_sum_total = np.zeros(n_outputs, dtype=np.float64)

        #
        self.sum_left = np.zeros(n_outputs, dtype=np.float64)
        #
        self.yt_sum_left = np.zeros(n_outputs, dtype=np.float64)
        self.y0_sum_left = np.zeros(n_outputs, dtype=np.float64)

        #
        self.sum_right = np.zeros(n_outputs, dtype=np.float64)
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
        self.sq_sum_total = 0.0
        #
        self.yt_sq_sum_total = 0.0
        self.y0_sq_sum_total = 0.0

        #
        memset(&self.sum_total[0], 0, self.n_outputs * sizeof(double))
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
            self.sum_total[0] += wt_y_ik
            #
            self.yt_sum_total[0] += wt_y_ik
            self.y0_sum_total[0] += w0_y_ik

            #
            self.sq_sum_total += wt_y_ik * y_ik
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
        memset(&self.sum_left[0], 0, n_bytes)
        #
        memset(&self.yt_sum_left[0], 0, n_bytes)
        memset(&self.y0_sum_left[0], 0, n_bytes)

        #
        memcpy(&self.sum_right[0], &self.sum_total, n_bytes)
        #
        memcpy(&self.yt_sum_right[0], &self.yt_sum_total[0], n_bytes)
        memcpy(&self.y0_sum_right[0], &self.y0_sum_total[0], n_bytes)

        #
        self.weighted_n_left = 0.0
        #
        self.nt_left = 0.0
        self.n0_left = 0.0

        #
        self.weighted_n_right = self.weighted_n_node_samples
        #
        self.nt_right = self.nt_total
        self.n0_right = self.n0_total

        self.pos = self.start
        return 0

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criterion at pos=end."""
        cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
        #
        memset(&self.sum_right[0], 0, n_bytes)
        #
        memset(&self.yt_sum_right[0], 0, n_bytes)
        memset(&self.y0_sum_right[0], 0, n_bytes)

        #
        memcpy(&self.sum_left[0], &self.sum_total[0], n_bytes)
        #
        memcpy(&self.yt_sum_left[0], &self.yt_sum_total[0], n_bytes)
        memcpy(&self.y0_sum_left[0], &self.y0_sum_total[0], n_bytes)

        #
        self.weighted_n_right = 0.0
        #
        self.nt_right = 0.0
        self.n0_right = 0.0

        #
        self.weighted_n_left = self.weighted_n_node_samples
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
                self.sum_left[0] += w * self.y[i, 0]
                #
                self.yt_sum_left[0] += w * self.y[i, 0]
                self.y0_sum_left[0] += (1.0 - w) * self.y[i, 0]

                #
                self.weighted_n_left += w
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
                self.sum_left[0] -= w * self.y[i, 0]
                #
                self.yt_sum_left[0] -= w * self.y[i, 0]
                self.y0_sum_left[0] -= (1.0 - w) * self.y[i, 0]

                #
                self.weighted_n_left -= w
                #
                self.nt_left -= w
                self.n0_left -= (1.0 - w)

        #
        self.weighted_n_right = (self.weighted_n_node_samples -
                                 self.weighted_n_left)
        #
        self.nt_right = (self.nt_total - self.nt_left)
        self.n0_right = (self.n0_total - self.n0_left)

        #
        self.sum_right[0] = self.sum_total - self.sum_left[0]
        #
        self.yt_sum_right[0] = self.yt_sum_total[0] - self.yt_sum_left[0]
        self.y0_sum_right[0] = self.y0_sum_total[0] - self.y0_sum_left[0]

        self.pos = new_pos
        return 0

    cdef double node_impurity(self) nogil:
        cdef double impurity
        impurity = (self.yt_sq_sum_total / (self.nt_total + eps) - (self.yt_sum_total / (self.nt_total + eps))**2.0) / (self.nt_total + eps)
        impurity += (self.y0_sq_sum_total / (self.n0_total + eps) - (self.y0_sum_total / (self.n0_total + eps))**2.0) / (self.n0_total + eps)
        impurity -= (self.yt_sum_total / (self.nt_total + eps) - self.y0_sum_total / (self.n0_total + eps))**2.0
        return impurity

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t pos = self.pos
        cdef SIZE_t start = self.start

        cdef DOUBLE_t y_ik

        #
        cdef double sq_sum_left = 0.0
        #
        cdef double yt_sq_sum_left = 0.0
        cdef double y0_sq_sum_left = 0.0

        #
        cdef double sq_sum_right
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
            sq_sum_left += w * y_ik * y_ik
            #
            yt_sq_sum_left += w * y_ik * y_ik
            y0_sq_sum_left += (1. - w) * y_ik * y_ik

        #
        sq_sum_right = self.sq_sum_total - sq_sum_left
        #
        yt_sq_sum_right = self.yt_sq_sum_total - yt_sq_sum_left
        y0_sq_sum_right = self.y0_sq_sum_total - y0_sq_sum_right

        #
        #impurity_left[0] = sq_sum_left / self.weighted_n_left
        #
        impurity_left[0] = (yt_sq_sum_left / (self.nt_left + eps) - (self.yt_sum_left[0] / (self.nt_left + eps))**2.0) / (self.nt_left + eps)
        impurity_left[0] += (y0_sq_sum_left / (self.n0_left + eps) - (self.y0_sum_left[0] / (self.n0_left + eps))**2.0) / (self.n0_left + eps)
        
        #
        #impurity_right[0] = sq_sum_right / self.weighted_n_right
        #
        impurity_right[0] = (yt_sq_sum_right / (self.nt_right + eps) - (self.yt_sum_right[0] / (self.nt_right + eps))**2.0) / (self.nt_right + eps)
        impurity_right[0] += (y0_sq_sum_right / (self.n0_right + eps) - (self.y0_sum_right[0] / (self.n0_right + eps))**2.0) / (self.n0_right + eps)
        
        #impurity_left[0] /= self.n_outputs
        #impurity_right[0] /= self.n_outputs
        impurity_left[0] -= (self.yt_sum_left[0] / (self.nt_left + eps) - self.y0_sum_left[0] / (self.n0_left + eps)) ** 2.0
        impurity_right[0] -= (self.yt_sum_right[0] / (self.nt_right + eps) - self.y0_sum_right[0] / (self.n0_right + eps)) ** 2.0
    
    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] into dest."""
        dest[0] = self.yt_sum_total / (self.nt_total + eps) - self.y0_sum_total /  (self.n0_total + eps)


    cdef double proxy_impurity_improvement(self) nogil:
        cdef double impurity_left
        cdef double impurity_right
        self.children_impurity(&impurity_left, &impurity_right)

        return (- (self.nt_right + self.n0_right) * impurity_right
                - (self.nt_left + self.n0_left) * impurity_left)

    cdef double impurity_improvement(self, double impurity_parent,
                                     double impurity_left,
                                     double impurity_right) nogil:
        return (impurity_parent - ((self.nt_right + self.n0_right) / (self.nt_total + self.n0_total) * impurity_right)
                                - ((self.nt_left + self.n0_left) / (self.nt_total + self.n0_total) * impurity_left))