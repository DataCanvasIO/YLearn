# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

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

cdef class PRegCriteria1(RegressionCriterion):
    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
        samples[start:end]. Specifically, we directly choose the max value of
        self.sum_total as the node impurity. In the end of the day, after 
        estimating the node value, we directly choose the index of the max value
        of self.sum_total as the index of the desired treatment.
        """
        cdef SIZE_t k
        cdef double max_sum = - INFINITY

        for k in range(self.n_outputs):
            if self.sum_total[k] > max_sum:
                max_sum = self.sum_total[k]

        return (alpha - max_sum / self.weighted_n_node_samples)

    cdef double proxy_impurity_improvement(self) nogil:
        """Compute a proxy of the impurity reduction.
        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.
        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.

        The PRegCriterion proxy is derived from
            max_{k} (sum_{i left}y_i) / N_left + max_{k} (sum_{i right}y_i) / N_right
        """
        cdef SIZE_t k
        cdef double proxy_impurity_left = -INFINITY
        cdef double proxy_impurity_right = -INFINITY

        for k in range(self.n_outputs):
            if self.sum_left[k] > proxy_impurity_left:
                proxy_impurity_left = self.sum_left[k]
            
            if self.sum_right[k] > proxy_impurity_right:
                proxy_impurity_right = self.sum_right[k]
        # ? not sure about this
        return (proxy_impurity_left / self.weighted_n_left +
                proxy_impurity_right / self.weighted_n_right)

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
        left child (samples[start:pos]) and the impurity the right child
        (samples[pos:end]).
        """
        cdef SIZE_t k
        cdef double max_sum_left = -INFINITY
        cdef double max_sum_right = -INFINITY

        for k in range(self.n_outputs):
            if self.sum_left[k] > max_sum_left:
                max_sum_left = self.sum_left[k]

            if self.sum_right[k] > max_sum_right:
                max_sum_right = self.sum_right[k]

        impurity_left[0] = alpha - max_sum_left / self.weighted_n_left
        impurity_right[0] = alpha - max_sum_right / self.weighted_n_right


cdef class PRegCriteria(RegressionCriterion):
    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
        samples[start:end]. Specifically, we directly choose the max value of
        self.sum_total as the node impurity. In the end of the day, after 
        estimating the node value, we directly choose the index of the max value
        of self.sum_total as the index of the desired treatment.
        """
        cdef SIZE_t k
        cdef double max_sum = - INFINITY

        for k in range(self.n_outputs):
            if self.sum_total[k] > max_sum:
                max_sum = self.sum_total[k]

        return (alpha - max_sum / self.weighted_n_node_samples)

    cdef double proxy_impurity_improvement(self) nogil:
        """Compute a proxy of the impurity reduction.
        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.
        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.

        The PRegCriterion proxy is derived from
            max_{k} (sum_{i left}y_i) / N_left + max_{k} (sum_{i right}y_i) / N_right
        """
        cdef SIZE_t k
        cdef double proxy_impurity_left = -INFINITY
        cdef double proxy_impurity_right = -INFINITY

        for k in range(self.n_outputs):
            if self.sum_left[k] > proxy_impurity_left:
                proxy_impurity_left = self.sum_left[k]
            
            if self.sum_right[k] > proxy_impurity_right:
                proxy_impurity_right = self.sum_right[k]
        # ? not sure about this
        #return (proxy_impurity_left / self.weighted_n_left +
        #        proxy_impurity_right / self.weighted_n_right)
        return (proxy_impurity_left + proxy_impurity_right)

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
        left child (samples[start:pos]) and the impurity the right child
        (samples[pos:end]).
        """
        cdef SIZE_t k
        cdef double max_sum_left = -INFINITY
        cdef double max_sum_right = -INFINITY

        for k in range(self.n_outputs):
            if self.sum_left[k] > max_sum_left:
                max_sum_left = self.sum_left[k]

            if self.sum_right[k] > max_sum_right:
                max_sum_right = self.sum_right[k]

        impurity_left[0] = alpha - max_sum_left / self.weighted_n_left
        impurity_right[0] = alpha - max_sum_right / self.weighted_n_right