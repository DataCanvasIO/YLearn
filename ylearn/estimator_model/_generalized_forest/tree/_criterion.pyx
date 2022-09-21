from libc.string cimport memcpy
from libc.string cimport memset
from libc.stdlib cimport calloc
from libc.stdlib cimport malloc

from libc.math cimport log
from libc.math cimport exp
from scipy.linalg.cython_lapack cimport dgelsy, dgetrf, dgetri, dgecon, dlacpy, dlange

import numpy as np
cimport numpy as np

np.import_array()

cdef double INFINITY = np.inf

rc = np.finfo(np.float64).eps
cdef inline double RCOND = rc

from ylearn.sklearn_ex.cloned.tree._criterion cimport Criterion

from ylearn.sklearn_ex.cloned.tree._tree cimport DOUBLE_t
from ylearn.sklearn_ex.cloned.tree._tree cimport SIZE_t

cdef class CriterionEx(Criterion):
    cdef int init_ex(self, const DOUBLE_t[:, ::1] y, const DOUBLE_t[:, ::1] treatment, DOUBLE_t* sample_weight, # fix
                  double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                  SIZE_t end) nogil except -1:
        """Placeholder for a method which will initialize the criterion.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        y : array-like, dtype=DOUBLE_t
            y is a buffer that can store values for n_outputs target variables
        treatment : array-like, dtype=DOUBLE_t
            treatment is a buffer that can store values for n_outputs treatment variables
        sample_weight : array-like, dtype=DOUBLE_t
            The weight of each sample
        weighted_n_samples : double
            The total weight of the samples being considered
        samples : array-like, dtype=SIZE_t
            Indices of the samples in X and y, where samples[start:end]
            correspond to the samples in this node
        start : SIZE_t
            The first sample to be used on this node
        end : SIZE_t
            The last sample used on this node

        """
        pass


cdef class GrfTreeCriterion(CriterionEx):
    #
    # def __cinit__(self, SIZE_t n_outputs, SIZE_t n_samples, SIZE_t d_tr):
    #     """Initialize parameters for this criterion.
    #     Parameters
    #     ----------
    #     n_outputs : SIZE_t
    #         The number of targets to be predicted
    #     n_samples : SIZE_t
    #         The total number of samples to fit on
    #     """
    #     # Default values
    #     self.sample_weight = NULL
    #
    #     self.samples = NULL
    #     self.start = 0
    #     self.pos = 0
    #     self.end = 0
    #
    #     self.n_outputs = n_outputs
    #     self.n_samples = n_samples
    #     self.d_tr = d_tr
    #     self.n_node_samples = 0
    #     self.weighted_n_node_samples = 0.0
    #     self.weighted_n_left = 0.0
    #     self.weighted_n_right = 0.0
    #
    #     self.sq_sum_total = 0.0
    #     self.sum_rho = 0.0
    #
    #     self.sum_total = np.zeros(n_outputs, dtype=np.float64)
    #     self.sum_left = np.zeros(n_outputs, dtype=np.float64)
    #     self.sum_right = np.zeros(n_outputs, dtype=np.float64)
    #
    #     self.grad = np.zeros([d_tr, d_tr], dtype=np.float64)
    #     self.sum_tr = np.zeros(d_tr, dtype=np.float64)
    #     self.mean_tr = np.zeros(d_tr, dtype=np.float64)
    #     self.mean_sum = np.zeros(n_outputs, dtype=np.float64)
    #
    #     self.rho = np.zeros(n_samples, dtype=np.float64)
    #     # self.sum_rho_total = np.zeros(d_tr, dtype=np.float64)
    #     # self.sum_rho_left = np.zeros(d_tr, dtype=np.float64)
    #     # self.sum_rho_right = np.zeros(d_tr, dtype=np.float64)
    #     self.sum_rho = 0.0
    #     self.sum_rho_left = 0.0
    #     self.sum_rho_right = 0.0
    #
    # def __reduce__(self):
    #     return (type(self), (self.n_outputs, self.n_samples, self.d_tr), self.__getstate__())

    cdef int init_ex(self, const DOUBLE_t[:, ::1] y, const DOUBLE_t[:, ::1] treatment, DOUBLE_t* sample_weight, # fix
                  double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                  SIZE_t end) nogil except -1:
        return 0

    # cdef int init(self, const DOUBLE_t[:, ::1] y, const DOUBLE_t[:, ::1] treatment, DOUBLE_t* sample_weight, #
    #               double weighted_n_samples, SIZE_t* samples, SIZE_t start,
    #               SIZE_t end) nogil except -1:
    #     """Initialize the criterion.
    #     This initializes the criterion at node samples[start:end] and children
    #     samples[start:start] and samples[start:end].
    #     """
    #     # Initialize fields
    #     self.y = y
    #     self.treatment = treatment
    #
    #     self.sample_weight = sample_weight
    #     self.samples = samples
    #     self.start = start
    #     self.end = end
    #     self.n_node_samples = end - start
    #     self.weighted_n_samples = weighted_n_samples
    #     self.weighted_n_node_samples = 0.
    #
    #     cdef SIZE_t i
    #     cdef SIZE_t p
    #     cdef SIZE_t k
    #     cdef SIZE_t k_tr
    #     cdef SIZE_t j_tr
    #
    #     cdef DOUBLE_t y_ik
    #     cdef DOUBLE_t tr_ik
    #     cdef DOUBLE_t tr_dif_ik
    #     cdef DOUBLE_t y_dif_ik
    #     cdef DOUBLE_t grad_coef_ik = 0.0
    #     cdef DOUBLE_t w_tr_ik
    #     cdef DOUBLE_t w_y_ik
    #     cdef DOUBLE_t w = 1.0
    #
    #     # create arrays for sotering y_node, y_dif, tr_node and tr_dif
    #     cdef int n_node, d_y, d_tr, ndy, ndtr, lda, ldb, info, rank, lwork, dytr, ndytr, dtrdtr
    #     cdef int grad_lda, grad_info, grad_lwork, grad_m
    #     cdef double rcond
    #     cdef int* jpvt, grad_pvt
    #     cdef double* work, grad_work
    #
    #     n_node = self.n_node_samples
    #     lda = n_node
    #     ldb = n_node
    #     grad_lda = self.d_tr
    #     grad_m = self.d_tr
    #     grad_lwork = self.d_tr * self.d_tr
    #     d_y = self.n_outputs
    #     d_tr = self.d_tr
    #     dytr = d_y * d_tr
    #     dtrdtr = d_tr * d_tr
    #     ndy = n_node * d_y
    #     ndtr = n_node * d_tr
    #     ndytr = n_node * dytr
    #
    #     work = <DOUBLE_t*> malloc(lwork * sizeof(DOUBLE_t))
    #     lwork = max(min(d_tr, n_node) + 3 * n_node + 1, 2 * min(n_node, d_tr) + d_y)
    #     jpvt = <int*> calloc(d_tr, sizeof(int))
    #     grad_pvt = <int*> malloc(d_tr * sizeof(int))
    #     grad_work = <DOUBLE_t*> malloc(grad_lwork * sizeof(DOUBLE_t))
    #     rcond = max(n_node, d_tr) * RCOND
    #
    #     cdef double y_node[ndy]
    #     cdef double tr_node[ndtr]
    #     cdef double ls_coef[dytr]
    #     # cdef double grad_coef[ndtr]
    #     cdef double *y_node=<double*> malloc(n_node * d_y * sizeof(double))
    #
    #     f_grad = <DOUBLE_t*> calloc(dtrdtr, sizeof(double)) # grad in Fortran-contiguous layout
    #     grad_coef = <DOUBLE_t*> calloc(ndtr, sizeof(double))
    #     y_dif = <DOUBLE_t*> calloc(ndy, sizeof(double))
    #     tr_dif = <DOUBLE_t*> calloc(ndtr, sizeof(double))
    #
    #     self.sq_sum_total = 0.0
    #     self.sum_rho = 0.0
    #
    #     memset(&self.sum_total[0], 0, self.n_outputs * sizeof(double))
    #     memset(&self.rho[0, 0], 0, self.n_samples * self.d_tr * sizeof(double))
    #     memset(&self.grad[0, 0], 0, self.d_tr * self.d_tr * sizeof(double))
    #     memset(&self.sum_tr[0], 0, self.d_tr * sizeof(double))
    #     memset(&self.sum_rho_total[0], 0, self.d_tr * sizeof(double))
    #     memset(&self.mean_sum[0], 0, self.n_outputs * sizeof(double))
    #     memset(&self.mean_tr[0], 0, self.d_tr * sizeof(double))
    #
    #     # TODO: Note that the following code computes the value of ls_coef by first copying
    #     # necessary info of self.treatment and self.y to new arrays, then calling the lapack
    #     for p in range(start, end):
    #         i = samples[p]
    #
    #         if sample_weight != NULL:
    #             w = sample_weight[i]
    #
    #         for k in range(self.n_outputs):
    #             y_ik = self.y[i, k]
    #             w_y_ik = w * y_ik
    #             y_node[p + n * k] = w_y_ik
    #             self.sum_total[k] += w_y_ik
    #             self.sq_sum_total += w_y_ik * y_ik
    #
    #         # compute sum of tr for computing its mean which then gives us tr_dif
    #         for k_tr in range(self.d_tr):
    #             tr_ik = self.treatment[i, k_tr]
    #             w_tr_ik = w * tr_ik
    #             tr_node[p + n * k_tr] = w_tr_ik
    #             self.sum_tr[k_tr] += w_tr_ik
    #
    #         self.weighted_n_node_samples += w
    #
    #     # -2: compute the mean of treatment
    #     for k_tr in range(self.d_tr):
    #         self.mean_tr[k_tr] += self.sum_tr[k_tr] / self.weighted_n_node_samples
    #
    #     # -1: compute the mean of outcome
    #     for k in range(self.n_outputs):
    #         self.mean_sum[k] += self.sum_total[k] / self.weighted_n_node_samples
    #
    #     # 0: compute ls_coef
    #     # y_node: shape (n_node_samples, n_outputs)
    #     # tr_node: shape (n_node_samples, d_tr)
    #     # solve lstsq ls_coef_node = argmin_{\beta} || y_node - tr_node \beta ||_2
    #     dgelsy(&n_node, &d_tr, &d_y, tr_node, &lda, y_node, &ldb, &jpvt[0], &rcond, &rank, &work[0], &lwork, &info)
    #     for k_tr in range(self.d_tr):
    #         for k in range(self.n_outputs):
    #             ls_coef[k_tr + k * d_tr] = y_node[k_tr + k * ldb]
    #     # TODO: need to free these varialbes? free(jpvt)
    #
    #     # 1: compute tr_dif and y_dif
    #     for p in range(start, end):
    #         for k_tr in range(self.d_tr):
    #             # tr_dif_ik = tr_node[p + n * k_tr] - self.mean_tr[k_tr]
    #             tr_dif[p + n * k_tr] = tr_node[p + n * k_tr] - self.mean_tr[k_tr]
    #
    #         for k in range(self.n_outputs):
    #             # TODO: Note that although here n_outputs is not 1, we actually assume it to be 1 to simplify the implementation, see
    #             # y_dif_ik = y_node[p + n * k] - self.mean_sum[k]
    #             # y_dif[p + n * k] = y_dif_ik
    #             # grad_coef_ik += y_dif_ik
    #             grad_coef_ik += y_node[p + n * k] - self.mean_sum[k]
    #             for k_tr in range(self.d_tr):
    #                 grad_coef_ik -= tr_dif[p + n * k_tr] * ls_coef[k_tr + k * d_tr]
    #
    #         # 2: compute grad_coef
    #         for k_tr in range(self.d_tr):
    #             tr_dif_ik = tr_node[p + n * k_tr] - self.mean_tr[k_tr]
    #             grad_coef[p + n * k_tr] = tr_dif_ik * grad_coef_ik
    #             for j_tr in range(self.d_tr):
    #                 self.grad[k_tr, j_tr] += tr_dif_ik * tr_dif[p + n * j_tr] / n_node
    #
    #     # 3: compute grad
    #     for k_tr in range(self.d_tr):
    #         for j_tr in range(self.d_tr):
    #             f_grad[k_tr + d_tr * j_tr] = self.grad[k_tr, j_tr]
    #
    #     # 4: compute the inverse of grad
    #     dgetrf(&grad_m, &grad_m, f_grad, &grad_lda, grad_pvt, &grad_info)
    #     dgetri(&grad_m, f_grad, &grad_lda, grad_pvt, grad_work, &grad_lwork, &grad_info)
    #
    #     # 5: compute rho
    #     for p in range(start, end):
    #         i = samples[p]
    #
    #         for k_tr in range(self.d_tr):
    #             for j_tr in range(self.d_tr):
    #                 self.rho[i] += f_grad[j_tr + d_tr * k_tr] * grad_coef[p + n * j_tr]
    #
    #         self.sum_rho += self.rho[i]
    #
    #     # Reset to pos=start
    #     self.reset()
    #     return 0
    #
    # cdef int reset(self) nogil except -1:
    #     """Reset the criterion at pos=start."""
    #     cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
    #     memset(&self.sum_left[0], 0, n_bytes)
    #     memcpy(&self.sum_right[0], &self.sum_total[0], n_bytes)
    #
    #     memset(&self.sum_rho_left, 0, sizeof(double))
    #     memcpy(&self.sum_rho_right, & self.sum_rho, sizeof(double))
    #
    #     self.weighted_n_left = 0.0
    #     self.weighted_n_right = self.weighted_n_node_samples
    #     self.pos = self.start
    #     return 0
    #
    # cdef int reverse_reset(self) nogil except -1:
    #     """Reset the criterion at pos=end."""
    #     cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
    #     memset(&self.sum_right[0], 0, n_bytes)
    #     memcpy(&self.sum_left[0], &self.sum_total[0], n_bytes)
    #
    #     self.weighted_n_right = 0.0
    #     self.weighted_n_left = self.weighted_n_node_samples
    #     self.pos = self.end
    #     return 0
    #
    # cdef int update(self, SIZE_t new_pos) nogil except -1:
    #     """Updated statistics by moving samples[pos:new_pos] to the left."""
    #     cdef double* sample_weight = self.sample_weight
    #     cdef SIZE_t* samples = self.samples
    #
    #     cdef SIZE_t pos = self.pos
    #     cdef SIZE_t end = self.end
    #     cdef SIZE_t i
    #     cdef SIZE_t p
    #     cdef SIZE_t k
    #     cdef DOUBLE_t w = 1.0
    #
    #     # Update statistics up to new_pos
    #     #
    #     # Given that
    #     #           sum_left[x] +  sum_right[x] = sum_total[x]
    #     # and that sum_total is known, we are going to update
    #     # sum_left from the direction that require the least amount
    #     # of computations, i.e. from pos to new_pos or from end to new_pos.
    #     if (new_pos - pos) <= (end - new_pos):
    #         for p in range(pos, new_pos):
    #             i = samples[p]
    #
    #             if sample_weight != NULL:
    #                 w = sample_weight[i]
    #
    #             for k in range(self.n_outputs):
    #                 self.sum_left[k] += w * self.y[i, k]
    #
    #             self.weighted_n_left += w
    #     else:
    #         self.reverse_reset()
    #
    #         for p in range(end - 1, new_pos - 1, -1):
    #             i = samples[p]
    #
    #             if sample_weight != NULL:
    #                 w = sample_weight[i]
    #
    #             for k in range(self.n_outputs):
    #                 self.sum_left[k] -= w * self.y[i, k]
    #
    #             self.weighted_n_left -= w
    #
    #     self.weighted_n_right = (self.weighted_n_node_samples -
    #                              self.weighted_n_left)
    #     for k in range(self.n_outputs):
    #         self.sum_right[k] = self.sum_total[k] - self.sum_left[k]
    #
    #     self.pos = new_pos
    #     return 0
    #
    # cdef double node_impurity(self) nogil:
    #     """Evaluate the impurity of the current node.
    #     Evaluate the MSE criterion as impurity of the current node,
    #     i.e. the impurity of samples[start:end]. The smaller the impurity the
    #     better.
    #     """
    #     cdef double impurity
    #     cdef SIZE_t k
    #
    #     impurity = self.sq_sum_total / self.weighted_n_node_samples
    #     for k in range(self.n_outputs):
    #         impurity -= (self.sum_total[k] / self.weighted_n_node_samples)**2.0
    #
    #     return impurity / self.n_outputs
    #
    # cdef void children_impurity(self, double* impurity_left,
    #                             double* impurity_right) nogil:
    #     """Evaluate the impurity in children nodes.
    #     i.e. the impurity of the left child (samples[start:pos]) and the
    #     impurity the right child (samples[pos:end]).
    #     """
    #     cdef DOUBLE_t* sample_weight = self.sample_weight
    #     cdef SIZE_t* samples = self.samples
    #     cdef SIZE_t pos = self.pos
    #     cdef SIZE_t start = self.start
    #
    #     cdef DOUBLE_t y_ik
    #
    #     cdef double sq_sum_left = 0.0
    #     cdef double sq_sum_right
    #
    #     cdef SIZE_t i
    #     cdef SIZE_t p
    #     cdef SIZE_t k
    #     cdef DOUBLE_t w = 1.0
    #
    #     for p in range(start, pos):
    #         i = samples[p]
    #
    #         if sample_weight != NULL:
    #             w = sample_weight[i]
    #
    #         for k in range(self.n_outputs):
    #             y_ik = self.y[i, k]
    #             sq_sum_left += w * y_ik * y_ik
    #
    #     sq_sum_right = self.sq_sum_total - sq_sum_left
    #
    #     impurity_left[0] = sq_sum_left / self.weighted_n_left
    #     impurity_right[0] = sq_sum_right / self.weighted_n_right
    #
    #     for k in range(self.n_outputs):
    #         impurity_left[0] -= (self.sum_left[k] / self.weighted_n_left) ** 2.0
    #         impurity_right[0] -= (self.sum_right[k] / self.weighted_n_right) ** 2.0
    #
    #     impurity_left[0] /= self.n_outputs
    #     impurity_right[0] /= self.n_outputs
    #
    # cdef void node_value(self, double* dest) nogil:
    #     """Compute the node value of samples[start:end] into dest."""
    #     cdef SIZE_t k
    #
    #     for k in range(self.n_outputs):
    #         dest[k] = self.sum_total[k] / self.weighted_n_node_samples
    #
    # cdef double proxy_impurity_improvement(self) nogil:
    #     # TODO: update this expression
    #     """Compute a proxy of the impurity reduction.
    #     This method is used to speed up the search for the best split.
    #     It is a proxy quantity such that the split that maximizes this value
    #     also maximizes the impurity improvement. It neglects all constant terms
    #     of the impurity decrease for a given split.
    #     The absolute impurity improvement is only computed by the
    #     impurity_improvement method once the best split has been found.
    #     The MSE proxy is derived from
    #         sum_{i left}(y_i - y_pred_L)^2 + sum_{i right}(y_i - y_pred_R)^2
    #         = sum(y_i^2) - n_L * mean_{i left}(y_i)^2 - n_R * mean_{i right}(y_i)^2
    #     Neglecting constant terms, this gives:
    #         - 1/n_L * sum_{i left}(y_i)^2 - 1/n_R * sum_{i right}(y_i)^2
    #     """
    #     cdef SIZE_t k
    #     cdef double proxy_impurity_left = 0.0
    #     cdef double proxy_impurity_right = 0.0
    #
    #     for k in range(self.n_outputs):
    #         proxy_impurity_left += self.sum_left[k] * self.sum_left[k]
    #         proxy_impurity_right += self.sum_right[k] * self.sum_right[k]
    #
    #     return (proxy_impurity_left / self.weighted_n_left +
    #             proxy_impurity_right / self.weighted_n_right)
