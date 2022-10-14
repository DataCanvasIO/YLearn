# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.stdio cimport printf

from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.string cimport memset
from libc.stdlib cimport calloc
from libc.stdlib cimport malloc

from libc.math cimport log
# from scipy.linalg.cython_lapack cimport dgelsy, dgetrf, dgetri, dlacpy
from ._util cimport init_criterion

import numpy as np
cimport numpy as np

np.import_array()

cdef double INFINITY = np.inf

rc = np.finfo(np.float64).eps
cdef double EPS = 1e10
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


# TODO: considering change lines which use self.n_node_samples to self.weighted_n_node_samples
cdef class GrfTreeCriterion(CriterionEx):
    def __cinit__(self, SIZE_t n_outputs, SIZE_t n_samples, SIZE_t d_tr):
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
        self.d_tr = d_tr
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        self.sq_sum_total = 0.0
        
        self.sum_total = np.zeros(n_outputs, dtype=np.float64)
        self.sum_left = np.zeros(n_outputs, dtype=np.float64)
        self.sum_right = np.zeros(n_outputs, dtype=np.float64)

        self.grad = np.zeros([d_tr, d_tr], dtype=np.float64)
        self.sum_tr = np.zeros(d_tr, dtype=np.float64)
        self.mean_tr = np.zeros(d_tr, dtype=np.float64)
        self.mean_sum = np.zeros(n_outputs, dtype=np.float64)
        
        self.rho = np.zeros(n_samples, dtype=np.float64)
        self.sum_rho = 0.0
        self.sum_rho_left = 0.0
        self.sum_rho_right = 0.0

    def __reduce__(self):
        return (type(self), (self.n_outputs, self.n_samples, self.d_tr), self.__getstate__())

    cdef int init_ex(self, const DOUBLE_t[:, ::1] y, const DOUBLE_t[:, ::1] treatment, DOUBLE_t* sample_weight,
                  double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                  SIZE_t end) nogil except -1:
        """Initialize the criterion.
        This initializes the criterion at node samples[start:end] and children
        samples[start:start] and samples[start:end].
        """
        # Initialize fields
        self.y = y
        self.treatment = treatment

        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.

        self.sq_sum_total = 0.0
        self.sum_rho = 0.0

        init_criterion(
            self.n_outputs, self.d_tr, self.n_samples,
            &y[0, 0], &treatment[0,0],
            sample_weight, samples,
            start, end,
            &self.sum_total[0], &self.mean_sum[0],
            &self.sum_tr[0], &self.mean_tr[0],
            &self.rho[0], &self.grad[0, 0],
            &self.weighted_n_node_samples,
            &self.sum_rho
        )

        # cdef SIZE_t i
        # cdef SIZE_t p
        # cdef SIZE_t idx
        # cdef SIZE_t k
        # cdef SIZE_t k_tr
        # cdef SIZE_t j_tr
        #
        # cdef DOUBLE_t y_ik
        # cdef DOUBLE_t tr_ik
        # cdef DOUBLE_t tr_dif_ik
        # cdef DOUBLE_t y_dif_ik
        # cdef DOUBLE_t grad_coef_ik = 0.0
        # cdef DOUBLE_t w_tr_ik
        # cdef DOUBLE_t w_y_ik
        # cdef DOUBLE_t w = 1.0
        #
        # # create arrays for sotering y_node, y_dif, tr_node and tr_dif
        # cdef int lda, ldb, info, rank, lwork, grad_lwork
        # cdef char* UPLO = 'A' # TODO: watch out
        # cdef int n_, d_tr, d_y
        # cdef int grad_lda, grad_info, grad_m
        # cdef double rcond
        # cdef int* jpvt
        # cdef int* grad_pvt
        # cdef double* work
        # cdef double* grad_work
        #
        # cdef double rho_i_
        #
        # lda = self.n_node_samples
        # ldb = self.n_node_samples
        # grad_lda = self.d_tr
        # grad_m = self.d_tr
        # grad_lwork = self.d_tr * self.d_tr
        #
        # n_ = self.n_node_samples
        # d_tr = self.d_tr
        # d_y = self.n_outputs
        #
        # lwork = max(min(self.d_tr, self.n_node_samples) + 3 * self.d_tr + 1,
        #             2 * min(self.n_node_samples, self.d_tr) + self.n_outputs)
        # work = <DOUBLE_t*> malloc(lwork * sizeof(DOUBLE_t))
        #
        # jpvt = <int*> calloc(self.d_tr, sizeof(int))
        # grad_pvt = <int*> malloc(self.d_tr * sizeof(int))
        #
        # grad_work = <DOUBLE_t*> malloc(grad_lwork * sizeof(DOUBLE_t))
        # rcond = max(self.n_node_samples, self.d_tr) * RCOND
        #
        # cdef double* y_node = <DOUBLE_t*> malloc(self.n_node_samples * self.n_outputs * sizeof(DOUBLE_t))
        # cdef double* tr_node = <DOUBLE_t*> malloc(self.n_node_samples * self.d_tr * sizeof(DOUBLE_t))
        #
        # cdef double* ls_coef = <DOUBLE_t*> calloc(self.n_outputs * self.d_tr, sizeof(DOUBLE_t))
        # cdef double* f_grad = <DOUBLE_t*> calloc(self.d_tr * self.d_tr, sizeof(DOUBLE_t)) # grad in Fortran-contiguous layout
        # cdef double* grad_coef = <DOUBLE_t*> calloc(self.n_node_samples * self.d_tr, sizeof(DOUBLE_t))
        # cdef double* y_dif = <DOUBLE_t*> calloc(self.n_node_samples * self.n_outputs, sizeof(DOUBLE_t))
        # cdef double* tr_dif = <DOUBLE_t*> calloc(self.n_node_samples * self.d_tr, sizeof(DOUBLE_t))
        #
        # self.sq_sum_total = 0.0
        # self.sum_rho = 0.0
        #
        # memset(&self.sum_total[0], 0, self.n_outputs * sizeof(double))
        # memset(&self.rho[0], 0, self.n_samples * sizeof(double))
        # memset(&self.grad[0, 0], 0, self.d_tr * self.d_tr * sizeof(double))
        # memset(&self.sum_tr[0], 0, self.d_tr * sizeof(double))
        # memset(&self.mean_sum[0], 0, self.n_outputs * sizeof(double))
        # memset(&self.mean_tr[0], 0, self.d_tr * sizeof(double))
        #
        # # TODO: Note that the following code computes the value of ls_coef by first copying
        # # necessary info of self.treatment and self.y to new arrays, then calling the lapack
        #
        # for p in range(start, end):
        #     i = samples[p]
        #     idx = p - start
        #
        #     if sample_weight != NULL:
        #         w = sample_weight[i]
        #
        #     for k in range(self.n_outputs):
        #         y_ik = self.y[i, k]
        #         w_y_ik = w * y_ik
        #         y_node[idx + self.n_node_samples * k] = w_y_ik
        #         self.sum_total[k] += w_y_ik
        #         # self.sq_sum_total += w_y_ik * y_ik
        #
        #     # compute sum of tr for computing its mean which then gives us tr_dif
        #     for k_tr in range(self.d_tr):
        #         tr_ik = self.treatment[i, k_tr]
        #         w_tr_ik = w * tr_ik
        #         tr_node[idx + self.n_node_samples * k_tr] = w_tr_ik
        #         self.sum_tr[k_tr] += w_tr_ik
        #
        #     self.weighted_n_node_samples += w
        #
        # # -2: compute the mean of treatment
        # for k_tr in range(self.d_tr):
        #     self.mean_tr[k_tr] += self.sum_tr[k_tr] / self.weighted_n_node_samples
        #
        # # -1: compute the mean of outcome
        # for k in range(self.n_outputs):
        #     self.mean_sum[k] += self.sum_total[k] / self.weighted_n_node_samples
        #
        # # 0: compute ls_coef
        # # y_node: shape (n_node_samples, n_outputs)
        # # tr_node: shape (n_node_samples, d_tr)
        # # solve lstsq ls_coef_node = argmin_{\beta} || y_node - tr_node \beta ||_2
        # cdef double* y_node_cpy = <DOUBLE_t*> calloc(ldb * self.n_outputs, sizeof(DOUBLE_t))
        # cdef double* tr_node_cpy = <DOUBLE_t*> calloc(lda * self.d_tr, sizeof(DOUBLE_t))
        # dlacpy(UPLO, &lda, &d_tr, tr_node, &lda, tr_node_cpy, &lda)
        # dlacpy(UPLO, &ldb, &d_y, y_node, &ldb, y_node_cpy, &ldb)
        # dgelsy(&n_, &d_tr, &d_y, tr_node_cpy, &lda, y_node_cpy, &ldb,
        #        &jpvt[0], &rcond, &rank, &work[0], &lwork, &info)
        # # dgelsy(&n_, &d_tr, &d_y, tr_node, &lda, y_node, &ldb, &jpvt[0], &rcond, &rank, &work[0], &lwork, &info)
        # for k_tr in range(self.d_tr):
        #     for k in range(self.n_outputs):
        #         ls_coef[k_tr + k * self.d_tr] = y_node_cpy[k_tr + k * ldb]
        #
        # # 1: compute tr_dif and y_dif
        # for p in range(start, end):
        #     idx = p - start
        #
        #     for k_tr in range(self.d_tr):
        #         # tr_dif_ik = tr_node[p + n * k_tr] - self.mean_tr[k_tr]
        #         tr_dif[idx + self.n_node_samples * k_tr] = (tr_node[idx + self.n_node_samples * k_tr] -
        #                                                   self.mean_tr[k_tr])
        #
        #     for k in range(self.n_outputs):
        #         # TODO: Note that although here n_outputs is not 1, we actually assume it to be 1 to simplify the implementation, see
        #         # y_dif_ik = y_node[p + n * k] - self.mean_sum[k]
        #         # y_dif[p + n * k] = y_dif_ik
        #         # grad_coef_ik += y_dif_ik
        #         grad_coef_ik = y_node[idx + self.n_node_samples * k] - self.mean_sum[k]
        #         for k_tr in range(self.d_tr):
        #             grad_coef_ik -= tr_dif[idx + self.n_node_samples * k_tr] * ls_coef[k_tr + k * self.d_tr]
        #
        #     # # 2: compute grad_coef
        #     for k_tr in range(self.d_tr):
        #         tr_dif_ik = tr_dif[idx + self.n_node_samples * k_tr]
        #         grad_coef[idx + self.n_node_samples * k_tr] = tr_dif_ik * grad_coef_ik
        #
        #         for j_tr in range(self.d_tr):
        #             self.grad[k_tr, j_tr] += (tr_dif_ik * tr_dif[idx + self.n_node_samples * j_tr] /
        #                                       self.n_node_samples)
        #
        #
        # # 3: compute grad
        # # TODO: grad does not change after calling this line
        # for k_tr in range(self.d_tr):
        #     for j_tr in range(self.d_tr):
        #         f_grad[k_tr + self.d_tr * j_tr] = self.grad[k_tr, j_tr]
        #
        # for k_tr in range(self.d_tr):
        #     f_grad[k_tr + self.d_tr * k_tr] += 1e-7
        #
        # # 4: compute the inverse of grad
        # dgetrf(&grad_m, &grad_m, f_grad, &grad_lda, grad_pvt, &grad_info)
        # dgetri(&grad_m, f_grad, &grad_lda, grad_pvt, grad_work, &grad_lwork, &grad_info)
        #
        # # 5: compute rho
        # for p in range(start, end):
        #     i = samples[p]
        #     idx = p - start
        #
        #     if sample_weight != NULL:
        #             w = sample_weight[i]
        #
        #     for k_tr in range(self.d_tr):
        #         for j_tr in range(self.d_tr):
        #             self.rho[i] += (f_grad[j_tr + self.d_tr * k_tr] *
        #                             grad_coef[idx + self.n_node_samples * j_tr])
        #
        #     self.sum_rho += w * self.rho[i]
        #
        # free(y_node)
        # free(tr_node)
        # free(ls_coef)
        # free(f_grad)
        # free(grad_coef)
        # free(y_dif)
        # free(tr_dif)
        # free(grad_work)
        # free(grad_pvt)
        # free(jpvt)
        # free(work)
        # free(y_node_cpy)
        # free(tr_node_cpy)
        
        # Reset to pos=start
        self.reset()
        return 0

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start."""
        cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
        memset(&self.sum_left[0], 0, n_bytes)
        memcpy(&self.sum_right[0], &self.sum_total[0], n_bytes)

        # memset(&self.sum_rho_left, 0, sizeof(double))
        # memcpy(&self.sum_rho_right, &self.sum_rho, sizeof(double))
        self.sum_rho_left = 0
        self.sum_rho_right = self.sum_rho

        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples
        self.pos = self.start
        return 0

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criterion at pos=end."""
        cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
        memset(&self.sum_right[0], 0, n_bytes)
        memcpy(&self.sum_left[0], &self.sum_total[0], n_bytes)

        # memset(&self.sum_rho_right, 0, sizeof(double))
        # memcpy(&self.sum_rho_left, &self.sum_rho, sizeof(double))
        self.sum_rho_right = 0
        self.sum_rho_left = self.sum_rho

        self.weighted_n_right = 0.0
        self.weighted_n_left = self.weighted_n_node_samples
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
                    w = sample_weight[i]
                
                self.sum_rho_left += w * self.rho[i]

                self.weighted_n_left += w
        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]
                
                self.sum_rho_left -= w * self.rho[i]

                self.weighted_n_left -= w

        self.weighted_n_right = (self.weighted_n_node_samples -
                                 self.weighted_n_left)
        
        self.sum_rho_right = self.sum_rho - self.sum_rho_left

        self.pos = new_pos
        return 0

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node.
        Evaluate the MSE criterion as impurity of the current node,
        i.e. the impurity of samples[start:end]. The smaller the impurity the
        better.
        """
        # cdef double impurity_left = 0.0
        # cdef double impurity_right = 0.0

        # impurity_left += self.sum_rho_left * self.sum_rho_left
        # impurity_right += self.sum_rho_right * self.sum_rho_right
        
        # cdef double impurity = 0.0
        # impurity += (self.sum_rho * self.sum_rho) / self.weighted_n_node_samples
        return 1.0

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

        cdef double sum_rho_left = 0.0
        cdef double sum_rho_right

        cdef SIZE_t i
        cdef SIZE_t p
        cdef DOUBLE_t w = 1.0            

        for p in range(start, pos):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]
            
            sum_rho_left += w * self.rho[i]

        sum_rho_right = self.sum_rho - sum_rho_left

        impurity_left[0] = (sum_rho_left * sum_rho_left) / self.weighted_n_left
        impurity_right[0] = (sum_rho_right * sum_rho_right) / self.weighted_n_right


    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] into dest."""
        cdef SIZE_t k

        for k in range(self.n_outputs):
            dest[k] = self.sum_total[k] / self.weighted_n_node_samples

    cdef double proxy_impurity_improvement(self) nogil:
        """Compute a proxy of the impurity reduction.
        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.
        For our GrfCriterion, the proxy impurity is computed as
            \sum_{j = 1}^2 (n_j)^{-1} (sum_rho_j)^2
        where j can taken as left and right.
        """
        cdef double proxy_impurity_left = 0.0
        cdef double proxy_impurity_right = 0.0

        proxy_impurity_left += self.sum_rho_left * self.sum_rho_left
        proxy_impurity_right += self.sum_rho_right * self.sum_rho_right

        return (proxy_impurity_left / self.weighted_n_left +
                proxy_impurity_right / self.weighted_n_right)

    cdef double impurity_improvement(self, double impurity_parent,
                                     double impurity_left,
                                     double impurity_right) nogil:
        """Compute the improvement in impurity.

        This method computes the improvement in impurity when a split occurs.
        The weighted impurity improvement equation is the following:

            right_impurity_improvement + left_impurity_impurity_improvement

        where N is the total number of samples, N_t is the number of samples
        at the current node, N_t_L is the number of samples in the left child,
        and N_t_R is the number of samples in the right child,

        Parameters
        ----------
        impurity_parent : double
            The initial impurity of the parent node before the split

        impurity_left : double
            The impurity of the left child

        impurity_right : double
            The impurity of the right child

        Return
        ------
        double : improvement in impurity after the split occurs
        """
        # return (self.sum_rho_left * self.sum_rho_left / self.weighted_n_left
        #         + self.sum_rho_right * self.sum_rho_right / self.weighted_n_right)
        return impurity_left + impurity_right

# cdef class TESTGrfTreeCriterion(CriterionEx):
#     def __cinit__(self, SIZE_t n_outputs, SIZE_t n_samples, SIZE_t d_tr):
#         """Initialize parameters for this criterion.
#         Parameters
#         ----------
#         n_outputs : SIZE_t
#             The number of targets to be predicted
#         n_samples : SIZE_t
#             The total number of samples to fit on
#         """
#         # Default values
#         self.sample_weight = NULL
#
#         self.samples = NULL
#         self.start = 0
#         self.pos = 0
#         self.end = 0
#
#         self.n_outputs = n_outputs
#         self.n_samples = n_samples
#         self.d_tr = d_tr
#         self.n_node_samples = 0
#         self.weighted_n_node_samples = 0.0
#         self.weighted_n_left = 0.0
#         self.weighted_n_right = 0.0
#
#         self.sq_sum_total = 0.0
#
#         self.sum_total = np.zeros(n_outputs, dtype=np.float64)
#         self.sum_left = np.zeros(n_outputs, dtype=np.float64)
#         self.sum_right = np.zeros(n_outputs, dtype=np.float64)
#
#         self.grad = np.zeros([d_tr, d_tr], dtype=np.float64)
#         self.sum_tr = np.zeros(d_tr, dtype=np.float64)
#         self.mean_tr = np.zeros(d_tr, dtype=np.float64)
#         self.mean_sum = np.zeros(n_outputs, dtype=np.float64)
#
#         self.rho = np.zeros(n_samples, dtype=np.float64)
#         # self.sum_rho_total = np.zeros(d_tr, dtype=np.float64)
#         # self.sum_rho_left = np.zeros(d_tr, dtype=np.float64)
#         # self.sum_rho_right = np.zeros(d_tr, dtype=np.float64)
#         self.sum_rho = 0.0
#         self.sum_rho_left = 0.0
#         self.sum_rho_right = 0.0
#
#     def __reduce__(self):
#         return (type(self), (self.n_outputs, self.n_samples, self.d_tr), self.__getstate__())
#
#     cdef int init_ex(self, const DOUBLE_t[:, ::1] y, const DOUBLE_t[:, ::1] treatment, DOUBLE_t* sample_weight,
#                   double weighted_n_samples, SIZE_t* samples, SIZE_t start,
#                   SIZE_t end) nogil except -1:
#         """Initialize the criterion.
#         This initializes the criterion at node samples[start:end] and children
#         samples[start:start] and samples[start:end].
#         """
#         # Initialize fields
#         self.y = y
#         self.treatment = treatment
#
#         self.sample_weight = sample_weight
#         self.samples = samples
#         self.start = start
#         self.end = end
#         self.n_node_samples = end - start
#         self.weighted_n_samples = weighted_n_samples
#         self.weighted_n_node_samples = 0.
#
#         cdef SIZE_t i
#         cdef SIZE_t p
#         cdef SIZE_t idx
#         cdef SIZE_t k
#         cdef SIZE_t k_tr
#         cdef SIZE_t j_tr
#
#         cdef DOUBLE_t y_ik
#         cdef DOUBLE_t tr_ik
#         cdef DOUBLE_t tr_dif_ik
#         cdef DOUBLE_t y_dif_ik
#         cdef DOUBLE_t grad_coef_ik = 0.0
#         cdef DOUBLE_t w_tr_ik
#         cdef DOUBLE_t w_y_ik
#         cdef DOUBLE_t w = 1.0
#
#         # create arrays for sotering y_node, y_dif, tr_node and tr_dif
#         cdef int lda, ldb, info, rank, lwork, grad_lwork
#         cdef char* UPLO = 'A' # TODO: watch out
#         cdef int n_, d_tr, d_y # TODO: aug variables, may be removed
#         cdef int grad_lda, grad_info, grad_m
#         cdef double rcond
#         cdef int* jpvt
#         cdef int* grad_pvt
#         cdef double* work
#         cdef double* grad_work
#
#         cdef double rho_i_
#
#         lda = self.n_node_samples
#         ldb = self.n_node_samples
#         grad_lda = self.d_tr
#         grad_m = self.d_tr
#         grad_lwork = self.d_tr * self.d_tr
#
#         n_ = self.n_node_samples
#         d_tr = self.d_tr
#         d_y = self.n_outputs
#
#         lwork = max(min(self.d_tr, self.n_node_samples) + 3 * self.d_tr + 1,
#                     2 * min(self.n_node_samples, self.d_tr) + self.n_outputs)
#         work = <DOUBLE_t*> malloc(lwork * sizeof(DOUBLE_t))
#
#         jpvt = <int*> calloc(self.d_tr, sizeof(int))
#         grad_pvt = <int*> malloc(self.d_tr * sizeof(int))
#
#         grad_work = <DOUBLE_t*> malloc(grad_lwork * sizeof(DOUBLE_t))
#         rcond = max(self.n_node_samples, self.d_tr) * RCOND
#
#         cdef double* y_node = <DOUBLE_t*> malloc(self.n_node_samples * self.n_outputs * sizeof(DOUBLE_t))
#         cdef double* tr_node = <DOUBLE_t*> malloc(self.n_node_samples * self.d_tr * sizeof(DOUBLE_t))
#
#         cdef double* ls_coef = <DOUBLE_t*> calloc(self.n_outputs * self.d_tr, sizeof(DOUBLE_t))
#         cdef double* f_grad = <DOUBLE_t*> calloc(self.d_tr * self.d_tr, sizeof(DOUBLE_t)) # grad in Fortran-contiguous layout
#         cdef double* grad_coef = <DOUBLE_t*> calloc(self.n_node_samples * self.d_tr, sizeof(DOUBLE_t))
#         cdef double* y_dif = <DOUBLE_t*> calloc(self.n_node_samples * self.n_outputs, sizeof(DOUBLE_t))
#         cdef double* tr_dif = <DOUBLE_t*> calloc(self.n_node_samples * self.d_tr, sizeof(DOUBLE_t))
#
#         self.sq_sum_total = 0.0
#         self.sum_rho = 0.0
#
#         memset(&self.sum_total[0], 0, self.n_outputs * sizeof(double))
#         memset(&self.rho[0], 0, self.n_samples * sizeof(double))
#         memset(&self.grad[0, 0], 0, self.d_tr * self.d_tr * sizeof(double))
#         memset(&self.sum_tr[0], 0, self.d_tr * sizeof(double))
#         memset(&self.mean_sum[0], 0, self.n_outputs * sizeof(double))
#         memset(&self.mean_tr[0], 0, self.d_tr * sizeof(double))
#
#         # TODO: Note that the following code computes the value of ls_coef by first copying
#         # necessary info of self.treatment and self.y to new arrays, then calling the lapack
#
#         for p in range(start, end):
#             i = samples[p]
#             idx = p - start
#
#             if sample_weight != NULL:
#                 w = sample_weight[i]
#
#             for k in range(self.n_outputs):
#                 y_ik = self.y[i, k]
#                 w_y_ik = w * y_ik
#                 y_node[idx + self.n_node_samples * k] = w_y_ik
#                 self.sum_total[k] += w_y_ik
#                 self.sq_sum_total += w_y_ik * y_ik
#
#             # compute sum of tr for computing its mean which then gives us tr_dif
#             for k_tr in range(self.d_tr):
#                 tr_ik = self.treatment[i, k_tr]
#                 w_tr_ik = w * tr_ik
#                 tr_node[idx + self.n_node_samples * k_tr] = w_tr_ik
#                 self.sum_tr[k_tr] += w_tr_ik
#
#             self.weighted_n_node_samples += w
#
#         # -2: compute the mean of treatment
#         for k_tr in range(self.d_tr):
#             self.mean_tr[k_tr] += self.sum_tr[k_tr] / self.weighted_n_node_samples
#
#         # -1: compute the mean of outcome
#         for k in range(self.n_outputs):
#             self.mean_sum[k] += self.sum_total[k] / self.weighted_n_node_samples
#
#         # 0: compute ls_coef
#         # y_node: shape (n_node_samples, n_outputs)
#         # tr_node: shape (n_node_samples, d_tr)
#         # solve lstsq ls_coef_node = argmin_{\beta} || y_node - tr_node \beta ||_2
#         cdef double* y_node_cpy = <DOUBLE_t*> calloc(ldb * self.n_outputs, sizeof(DOUBLE_t))
#         cdef double* tr_node_cpy = <DOUBLE_t*> calloc(lda * self.d_tr, sizeof(DOUBLE_t))
#         dlacpy(UPLO, &lda, &d_tr, tr_node, &lda, tr_node_cpy, &lda)
#         dlacpy(UPLO, &ldb, &d_y, y_node, &ldb, y_node_cpy, &ldb)
#         dgelsy(&n_, &d_tr, &d_y, tr_node_cpy, &lda, y_node_cpy, &ldb,
#                &jpvt[0], &rcond, &rank, &work[0], &lwork, &info)
#         # dgelsy(&n_, &d_tr, &d_y, tr_node, &lda, y_node, &ldb, &jpvt[0], &rcond, &rank, &work[0], &lwork, &info)
#         for k_tr in range(self.d_tr):
#             for k in range(self.n_outputs):
#                 ls_coef[k_tr + k * self.d_tr] = y_node_cpy[k_tr + k * ldb]
#         # 1: compute tr_dif and y_dif
#         for p in range(start, end):
#             idx = p - start
#
#             for k_tr in range(self.d_tr):
#                 # tr_dif_ik = tr_node[p + n * k_tr] - self.mean_tr[k_tr]
#                 tr_dif[idx + self.n_node_samples * k_tr] = (tr_node[idx + self.n_node_samples * k_tr] -
#                                                           self.mean_tr[k_tr])
#
#             for k in range(self.n_outputs):
#                 # TODO: Note that although here n_outputs is not 1, we actually assume it to be 1 to simplify the implementation, see
#                 # y_dif_ik = y_node[p + n * k] - self.mean_sum[k]
#                 # y_dif[p + n * k] = y_dif_ik
#                 # grad_coef_ik += y_dif_ik
#                 grad_coef_ik = y_node[idx + self.n_node_samples * k] - self.mean_sum[k]
#                 for k_tr in range(self.d_tr):
#                     grad_coef_ik -= tr_dif[idx + self.n_node_samples * k_tr] * ls_coef[k_tr + k * self.d_tr]
#
#             # # 2: compute grad_coef
#             for k_tr in range(self.d_tr):
#                 tr_dif_ik = tr_dif[idx + self.n_node_samples * k_tr]
#                 grad_coef[idx + self.n_node_samples * k_tr] = tr_dif_ik * grad_coef_ik
#
#                 for j_tr in range(self.d_tr):
#                     self.grad[k_tr, j_tr] += (tr_dif_ik * tr_dif[idx + self.n_node_samples * j_tr] /
#                                               self.n_node_samples)
#
#
#         # 3: compute grad
#         # TODO: grad does not change after calling this line
#         for k_tr in range(self.d_tr):
#             for j_tr in range(self.d_tr):
#                 f_grad[k_tr + self.d_tr * j_tr] = self.grad[k_tr, j_tr]
#
#         for k_tr in range(self.d_tr):
#             f_grad[k_tr + self.d_tr * k_tr] += 1e-7
#
#         # 4: compute the inverse of grad
#         # TODO:  this line is correct
#         dgetrf(&grad_m, &grad_m, f_grad, &grad_lda, grad_pvt, &grad_info)
#         dgetri(&grad_m, f_grad, &grad_lda, grad_pvt, grad_work, &grad_lwork, &grad_info)
#
#         # 5: compute rho
#         for p in range(start, end):
#             i = samples[p]
#             idx = p - start
#
#             if sample_weight != NULL:
#                     w = sample_weight[i]
#
#             for k_tr in range(self.d_tr):
#                 for j_tr in range(self.d_tr):
#
#                     self.rho[i] += (f_grad[j_tr + self.d_tr * k_tr] *
#                                     grad_coef[idx + self.n_node_samples * j_tr])
#
#             self.sum_rho += w * self.rho[i]
#
#         free(y_node)
#         free(tr_node)
#         free(ls_coef)
#         free(f_grad)
#         free(grad_coef)
#         free(y_dif)
#         free(tr_dif)
#         free(grad_work)
#         free(grad_pvt)
#         free(jpvt)
#         free(work)
#         free(y_node_cpy)
#         free(tr_node_cpy)
#
#         # Reset to pos=start
#         self.reset()
#         return 0
#
#     cdef int reset(self) nogil except -1:
#         """Reset the criterion at pos=start."""
#         # ORIGINAL
#         cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
#         memset(&self.sum_left[0], 0, n_bytes)
#         memcpy(&self.sum_right[0], &self.sum_total[0], n_bytes)
#
#         memset(&self.sum_rho_left, 0, sizeof(double))
#         memcpy(&self.sum_rho_right, &self.sum_rho, sizeof(double))
#
#         self.weighted_n_left = 0.0
#         self.weighted_n_right = self.weighted_n_node_samples
#         self.pos = self.start
#         return 0
#
#         #TEST
#         # cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
#         # memset(&self.sum_left[0], 0, n_bytes)
#         # memcpy(&self.sum_right[0], &self.sum_total[0], n_bytes)
#
#         # self.weighted_n_left = 0.0
#         # self.weighted_n_right = self.weighted_n_node_samples
#         # self.pos = self.start
#         # return 0
#
#     cdef int reverse_reset(self) nogil except -1:
#         """Reset the criterion at pos=end."""
#         cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
#         memset(&self.sum_right[0], 0, n_bytes)
#         memcpy(&self.sum_left[0], &self.sum_total[0], n_bytes)
#
#         memset(&self.sum_rho_right, 0, sizeof(double))
#         memcpy(&self.sum_rho_left, &self.sum_rho, sizeof(double))
#
#         self.weighted_n_right = 0.0
#         self.weighted_n_left = self.weighted_n_node_samples
#         self.pos = self.end
#         return 0
#
#
#         #TEST
#         # cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
#         # memset(&self.sum_right[0], 0, n_bytes)
#         # memcpy(&self.sum_left[0], &self.sum_total[0], n_bytes)
#
#         # self.weighted_n_right = 0.0
#         # self.weighted_n_left = self.weighted_n_node_samples
#         # self.pos = self.end
#         # return 0
#
#     cdef int update(self, SIZE_t new_pos) nogil except -1:
#         """Updated statistics by moving samples[pos:new_pos] to the left."""
#         cdef double* sample_weight = self.sample_weight
#         cdef SIZE_t* samples = self.samples
#
#         cdef SIZE_t pos = self.pos
#         cdef SIZE_t end = self.end
#         cdef SIZE_t i
#         cdef SIZE_t p
#         cdef SIZE_t k
#         cdef DOUBLE_t w = 1.0
#         # PASS
#         # # Update statistics up to new_pos
#
#         # Given that
#         #           sum_left[x] +  sum_right[x] = sum_total[x]
#         # and that sum_total is known, we are going to update
#         # sum_left from the direction that require the least amount
#         # of computations, i.e. from pos to new_pos or from end to new_pos.
#         if (new_pos - pos) <= (end - new_pos):
#             for p in range(pos, new_pos):
#                 i = samples[p]
#
#                 if sample_weight != NULL:
#                     w = sample_weight[i]
#
#                 self.sum_rho_left += w * self.rho[i]
#
#                 # for k in range(self.n_outputs):
#                 #     self.sum_left[k] += w * self.y[i, k]
#
#                 self.weighted_n_left += w
#         else:
#             self.reverse_reset()
#
#             for p in range(end - 1, new_pos - 1, -1):
#                 i = samples[p]
#
#                 if sample_weight != NULL:
#                     w = sample_weight[i]
#
#                 self.sum_rho_left -= w * self.rho[i]
#
#                 # for k in range(self.n_outputs):
#                 #     self.sum_left[k] -= w * self.y[i, k]
#
#                 self.weighted_n_left -= w
#
#         self.weighted_n_right = (self.weighted_n_node_samples -
#                                  self.weighted_n_left)
#
#         self.sum_rho_right = self.sum_rho - self.sum_rho_left
#         # for k in range(self.n_outputs):
#         #     self.sum_right[k] = self.sum_total[k] - self.sum_left[k]
#
#         self.pos = new_pos
#         return 0
#
#         #TEST
#         # cdef double* sample_weight = self.sample_weight
#         # cdef SIZE_t* samples = self.samples
#
#         # cdef SIZE_t pos = self.pos
#         # cdef SIZE_t end = self.end
#         # cdef SIZE_t i
#         # cdef SIZE_t p
#         # cdef SIZE_t k
#         # cdef DOUBLE_t w = 1.0
#
#         # # Update statistics up to new_pos
#         # #
#         # # Given that
#         # #           sum_left[x] +  sum_right[x] = sum_total[x]
#         # # and that sum_total is known, we are going to update
#         # # sum_left from the direction that require the least amount
#         # # of computations, i.e. from pos to new_pos or from end to new_pos.
#         # if (new_pos - pos) <= (end - new_pos):
#         #     for p in range(pos, new_pos):
#         #         i = samples[p]
#
#         #         if sample_weight != NULL:
#         #             w = sample_weight[i]
#
#         #         for k in range(self.n_outputs):
#         #             self.sum_left[k] += w * self.y[i, k]
#
#         #         self.weighted_n_left += w
#         # else:
#         #     self.reverse_reset()
#
#         #     for p in range(end - 1, new_pos - 1, -1):
#         #         i = samples[p]
#
#         #         if sample_weight != NULL:
#         #             w = sample_weight[i]
#
#         #         for k in range(self.n_outputs):
#         #             self.sum_left[k] -= w * self.y[i, k]
#
#         #         self.weighted_n_left -= w
#
#         # self.weighted_n_right = (self.weighted_n_node_samples -
#         #                          self.weighted_n_left)
#         # for k in range(self.n_outputs):
#         #     self.sum_right[k] = self.sum_total[k] - self.sum_left[k]
#
#         # self.pos = new_pos
#         # return 0
#
#     cdef double node_impurity(self) nogil:
#         """Evaluate the impurity of the current node.
#         Evaluate the MSE criterion as impurity of the current node,
#         i.e. the impurity of samples[start:end]. The smaller the impurity the
#         better.
#         """
#         # PASS
#         # cdef double impurity_left = 0.0
#         # cdef double impurity_right = 0.0
#
#         # impurity_left += self.sum_rho_left * self.sum_rho_left
#         # impurity_right += self.sum_rho_right * self.sum_rho_right
#
#         # cdef double impurity = 0.0
#         # impurity += (self.sum_rho * self.sum_rho) / self.weighted_n_node_samples
#         # return EPS - impurity
#         return 1.0
#
#         # Test
#         # cdef double impurity
#         # cdef SIZE_t k
#
#         # impurity = self.sq_sum_total / self.weighted_n_node_samples
#         # for k in range(self.n_outputs):
#         #     impurity -= (self.sum_total[k] / self.weighted_n_node_samples)**2.0
#
#         # return impurity / self.n_outputs
#
#
#     cdef void children_impurity(self, double* impurity_left,
#                                 double* impurity_right) nogil:
#         """Evaluate the impurity in children nodes.
#         i.e. the impurity of the left child (samples[start:pos]) and the
#         impurity the right child (samples[pos:end]).
#         """
#         # original
#         # cdef DOUBLE_t* sample_weight = self.sample_weight
#         # cdef SIZE_t* samples = self.samples
#         # cdef SIZE_t pos = self.pos
#         # cdef SIZE_t start = self.start
#
#         # cdef double sum_rho_left = 0.0
#         # cdef double sum_rho_right
#
#         # cdef SIZE_t i
#         # cdef SIZE_t p
#         # cdef DOUBLE_t w = 1.0
#
#         # for p in range(start, pos):
#         #     i = samples[p]
#
#         #     if sample_weight != NULL:
#         #         w = sample_weight[i]
#
#         #     sum_rho_left += w * self.rho[i]
#
#         # sum_rho_right = self.sum_rho - sum_rho_left
#
#         # impurity_left[0] = 1 / ((sum_rho_left * sum_rho_left) / self.weighted_n_left)
#         # impurity_right[0] = 1 / ((sum_rho_right * sum_rho_right) / self.weighted_n_right)
#         impurity_left[0] = 1.0
#         impurity_right[0] = 1.0
#         # impurity_left[0] = self.weighted_n_left / (self.sum_rho_left * self.sum_rho_left)
#         # impurity_right[0] = self.weighted_n_right / (self.sum_rho_right * self.sum_rho_right)
#
#         # Test
#         # cdef DOUBLE_t* sample_weight = self.sample_weight
#         # cdef SIZE_t* samples = self.samples
#         # cdef SIZE_t pos = self.pos
#         # cdef SIZE_t start = self.start
#
#         # cdef DOUBLE_t y_ik
#
#         # cdef double sq_sum_left = 0.0
#         # cdef double sq_sum_right
#
#         # cdef SIZE_t i
#         # cdef SIZE_t p
#         # cdef SIZE_t k
#         # cdef DOUBLE_t w = 1.0
#
#         # for p in range(start, pos):
#         #     i = samples[p]
#
#         #     if sample_weight != NULL:
#         #         w = sample_weight[i]
#
#         #     for k in range(self.n_outputs):
#         #         y_ik = self.y[i, k]
#         #         sq_sum_left += w * y_ik * y_ik
#
#         # sq_sum_right = self.sq_sum_total - sq_sum_left
#
#         # impurity_left[0] = sq_sum_left / self.weighted_n_left
#         # impurity_right[0] = sq_sum_right / self.weighted_n_right
#
#         # for k in range(self.n_outputs):
#         #     impurity_left[0] -= (self.sum_left[k] / self.weighted_n_left) ** 2.0
#         #     impurity_right[0] -= (self.sum_right[k] / self.weighted_n_right) ** 2.0
#
#         # impurity_left[0] /= self.n_outputs
#         # impurity_right[0] /= self.n_outputs
#
#     cdef void node_value(self, double* dest) nogil:
#         """Compute the node value of samples[start:end] into dest."""
#         # PASS
#         cdef SIZE_t k
#
#         for k in range(self.n_outputs):
#             dest[k] = self.sum_total[k] / self.weighted_n_node_samples
#
#     cdef double impurity_improvement(self, double impurity_parent,
#                                      double impurity_left,
#                                      double impurity_right) nogil:
#         """Compute the improvement in impurity.
#
#         This method computes the improvement in impurity when a split occurs.
#         The weighted impurity improvement equation is the following:
#
#             N_t / N * (impurity - N_t_R / N_t * right_impurity
#                                 - N_t_L / N_t * left_impurity)
#
#         where N is the total number of samples, N_t is the number of samples
#         at the current node, N_t_L is the number of samples in the left child,
#         and N_t_R is the number of samples in the right child,
#
#         Parameters
#         ----------
#         impurity_parent : double
#             The initial impurity of the parent node before the split
#
#         impurity_left : double
#             The impurity of the left child
#
#         impurity_right : double
#             The impurity of the right child
#
#         Return
#         ------
#         double : improvement in impurity after the split occurs
#         """
#         # return (self.sum_rho_left * self.sum_rho_left / self.weighted_n_left
#         #         + self.sum_rho_right * self.sum_rho_right / self.weighted_n_right)
#
#         cdef DOUBLE_t* sample_weight = self.sample_weight
#         cdef SIZE_t* samples = self.samples
#         cdef SIZE_t pos = self.pos
#         cdef SIZE_t start = self.start
#
#         cdef double sum_rho_left = 0.0
#         cdef double sum_rho_right
#
#         cdef SIZE_t i
#         cdef SIZE_t p
#         cdef DOUBLE_t w = 1.0
#
#         for p in range(start, pos):
#             i = samples[p]
#
#             if sample_weight != NULL:
#                 w = sample_weight[i]
#
#             sum_rho_left += w * self.rho[i]
#
#         sum_rho_right = self.sum_rho - sum_rho_left
#
#         return (sum_rho_left * sum_rho_left / self.weighted_n_left
#                 + sum_rho_right * sum_rho_right / self.weighted_n_right)
#
#
#     cdef double proxy_impurity_improvement(self) nogil:
#         """Compute a proxy of the impurity reduction.
#         This method is used to speed up the search for the best split.
#         It is a proxy quantity such that the split that maximizes this value
#         also maximizes the impurity improvement. It neglects all constant terms
#         of the impurity decrease for a given split.
#         For our GrfCriterion, the proxy impurity is computed as
#             \sum_{j = 1}^2 (n_j)^{-1} (sum_rho_j)^2
#         where j can taken as left and right.
#         """
#         cdef double proxy_impurity_left = 0.0
#         cdef double proxy_impurity_right = 0.0
#
#         proxy_impurity_left += self.sum_rho_left * self.sum_rho_left
#         proxy_impurity_right += self.sum_rho_right * self.sum_rho_right
#
#         return (proxy_impurity_left / self.weighted_n_left +
#                 proxy_impurity_right / self.weighted_n_right)
#
#         # cdef SIZE_t k
#         # cdef double proxy_impurity_left = 0.0
#         # cdef double proxy_impurity_right = 0.0
#
#         # for k in range(self.n_outputs):
#         #     proxy_impurity_left += self.sum_left[k] * self.sum_left[k]
#         #     proxy_impurity_right += self.sum_right[k] * self.sum_right[k]
#
#         # return (proxy_impurity_left / self.weighted_n_left +
#         #         proxy_impurity_right / self.weighted_n_right)
#
# cdef class TestMSE(CriterionEx):
#     r"""Abstract regression criterion.
#
#     This handles cases where the target is a continuous value, and is
#     evaluated by computing the variance of the target values left and right
#     of the split point. The computation takes linear time with `n_samples`
#     by using ::
#
#         var = \sum_i^n (y_i - y_bar) ** 2
#             = (\sum_i^n y_i ** 2) - n_samples * y_bar ** 2
#     """
#
#     def __cinit__(self, SIZE_t n_outputs, SIZE_t n_samples, SIZE_t d_tr):
#         """Initialize parameters for this criterion.
#
#         Parameters
#         ----------
#         n_outputs : SIZE_t
#             The number of targets to be predicted
#
#         n_samples : SIZE_t
#             The total number of samples to fit on
#         """
#         # Default values
#         self.sample_weight = NULL
#
#         self.samples = NULL
#         self.start = 0
#         self.pos = 0
#         self.end = 0
#
#         self.n_outputs = n_outputs
#         self.n_samples = n_samples
#         self.n_node_samples = 0
#         self.weighted_n_node_samples = 0.0
#         self.weighted_n_left = 0.0
#         self.weighted_n_right = 0.0
#
#         self.sq_sum_total = 0.0
#
#         self.sum_total = np.zeros(n_outputs, dtype=np.float64)
#         self.sum_left = np.zeros(n_outputs, dtype=np.float64)
#         self.sum_right = np.zeros(n_outputs, dtype=np.float64)
#
#     def __reduce__(self):
#         return (type(self), (self.n_outputs, self.n_samples), self.__getstate__())
#
#     cdef int init_ex(self, const DOUBLE_t[:, ::1] y, const DOUBLE_t[:, ::1] treatment, DOUBLE_t* sample_weight, # fix
#                   double weighted_n_samples, SIZE_t* samples, SIZE_t start,
#                   SIZE_t end) nogil except -1:
#         # Initialize fields
#         self.y = y
#         self.sample_weight = sample_weight
#         self.samples = samples
#         self.start = start
#         self.end = end
#         self.n_node_samples = end - start
#         self.weighted_n_samples = weighted_n_samples
#         self.weighted_n_node_samples = 0.
#
#         cdef SIZE_t i
#         cdef SIZE_t p
#         cdef SIZE_t k
#         cdef DOUBLE_t y_ik
#         cdef DOUBLE_t w_y_ik
#         cdef DOUBLE_t w = 1.0
#         self.sq_sum_total = 0.0
#         memset(&self.sum_total[0], 0, self.n_outputs * sizeof(double))
#
#         for p in range(start, end):
#             i = samples[p]
#
#             if sample_weight != NULL:
#                 w = sample_weight[i]
#
#             for k in range(self.n_outputs):
#                 y_ik = self.y[i, k]
#                 w_y_ik = w * y_ik
#                 self.sum_total[k] += w_y_ik
#                 self.sq_sum_total += w_y_ik * y_ik
#
#             self.weighted_n_node_samples += w
#
#         # Reset to pos=start
#         self.reset()
#         return 0
#
#     cdef int init(self, const DOUBLE_t[:, ::1] y, DOUBLE_t* sample_weight,
#                   double weighted_n_samples, SIZE_t* samples, SIZE_t start,
#                   SIZE_t end) nogil except -1:
#         """Initialize the criterion.
#
#         This initializes the criterion at node samples[start:end] and children
#         samples[start:start] and samples[start:end].
#         """
#         # Initialize fields
#         self.y = y
#         self.sample_weight = sample_weight
#         self.samples = samples
#         self.start = start
#         self.end = end
#         self.n_node_samples = end - start
#         self.weighted_n_samples = weighted_n_samples
#         self.weighted_n_node_samples = 0.
#
#         cdef SIZE_t i
#         cdef SIZE_t p
#         cdef SIZE_t k
#         cdef DOUBLE_t y_ik
#         cdef DOUBLE_t w_y_ik
#         cdef DOUBLE_t w = 1.0
#         self.sq_sum_total = 0.0
#         memset(&self.sum_total[0], 0, self.n_outputs * sizeof(double))
#
#         for p in range(start, end):
#             i = samples[p]
#
#             if sample_weight != NULL:
#                 w = sample_weight[i]
#
#             for k in range(self.n_outputs):
#                 y_ik = self.y[i, k]
#                 w_y_ik = w * y_ik
#                 self.sum_total[k] += w_y_ik
#                 self.sq_sum_total += w_y_ik * y_ik
#
#             self.weighted_n_node_samples += w
#
#         # Reset to pos=start
#         self.reset()
#         return 0
#
#     cdef int reset(self) nogil except -1:
#         """Reset the criterion at pos=start."""
#         cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
#         memset(&self.sum_left[0], 0, n_bytes)
#         memcpy(&self.sum_right[0], &self.sum_total[0], n_bytes)
#
#         self.weighted_n_left = 0.0
#         self.weighted_n_right = self.weighted_n_node_samples
#         self.pos = self.start
#         return 0
#
#     cdef int reverse_reset(self) nogil except -1:
#         """Reset the criterion at pos=end."""
#         cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
#         memset(&self.sum_right[0], 0, n_bytes)
#         memcpy(&self.sum_left[0], &self.sum_total[0], n_bytes)
#
#         self.weighted_n_right = 0.0
#         self.weighted_n_left = self.weighted_n_node_samples
#         self.pos = self.end
#         return 0
#
#     cdef int update(self, SIZE_t new_pos) nogil except -1:
#         """Updated statistics by moving samples[pos:new_pos] to the left."""
#         cdef double* sample_weight = self.sample_weight
#         cdef SIZE_t* samples = self.samples
#
#         cdef SIZE_t pos = self.pos
#         cdef SIZE_t end = self.end
#         cdef SIZE_t i
#         cdef SIZE_t p
#         cdef SIZE_t k
#         cdef DOUBLE_t w = 1.0
#
#         # Update statistics up to new_pos
#         #
#         # Given that
#         #           sum_left[x] +  sum_right[x] = sum_total[x]
#         # and that sum_total is known, we are going to update
#         # sum_left from the direction that require the least amount
#         # of computations, i.e. from pos to new_pos or from end to new_pos.
#         if (new_pos - pos) <= (end - new_pos):
#             for p in range(pos, new_pos):
#                 i = samples[p]
#
#                 if sample_weight != NULL:
#                     w = sample_weight[i]
#
#                 for k in range(self.n_outputs):
#                     self.sum_left[k] += w * self.y[i, k]
#
#                 self.weighted_n_left += w
#         else:
#             self.reverse_reset()
#
#             for p in range(end - 1, new_pos - 1, -1):
#                 i = samples[p]
#
#                 if sample_weight != NULL:
#                     w = sample_weight[i]
#
#                 for k in range(self.n_outputs):
#                     self.sum_left[k] -= w * self.y[i, k]
#
#                 self.weighted_n_left -= w
#
#         self.weighted_n_right = (self.weighted_n_node_samples -
#                                  self.weighted_n_left)
#         for k in range(self.n_outputs):
#             self.sum_right[k] = self.sum_total[k] - self.sum_left[k]
#
#         self.pos = new_pos
#         return 0
#
#     cdef void node_value(self, double* dest) nogil:
#         """Compute the node value of samples[start:end] into dest."""
#         cdef SIZE_t k
#
#         for k in range(self.n_outputs):
#             dest[k] = self.sum_total[k] / self.weighted_n_node_samples
#
#     cdef double node_impurity(self) nogil:
#         """Evaluate the impurity of the current node.
#
#         Evaluate the MSE criterion as impurity of the current node,
#         i.e. the impurity of samples[start:end]. The smaller the impurity the
#         better.
#         """
#         cdef double impurity
#         cdef SIZE_t k
#
#         impurity = self.sq_sum_total / self.weighted_n_node_samples
#         for k in range(self.n_outputs):
#             impurity -= (self.sum_total[k] / self.weighted_n_node_samples)**2.0
#
#         return impurity / self.n_outputs
#
#     cdef double proxy_impurity_improvement(self) nogil:
#         """Compute a proxy of the impurity reduction.
#
#         This method is used to speed up the search for the best split.
#         It is a proxy quantity such that the split that maximizes this value
#         also maximizes the impurity improvement. It neglects all constant terms
#         of the impurity decrease for a given split.
#
#         The absolute impurity improvement is only computed by the
#         impurity_improvement method once the best split has been found.
#
#         The MSE proxy is derived from
#
#             sum_{i left}(y_i - y_pred_L)^2 + sum_{i right}(y_i - y_pred_R)^2
#             = sum(y_i^2) - n_L * mean_{i left}(y_i)^2 - n_R * mean_{i right}(y_i)^2
#
#         Neglecting constant terms, this gives:
#
#             - 1/n_L * sum_{i left}(y_i)^2 - 1/n_R * sum_{i right}(y_i)^2
#         """
#         cdef SIZE_t k
#         cdef double proxy_impurity_left = 0.0
#         cdef double proxy_impurity_right = 0.0
#
#         for k in range(self.n_outputs):
#             proxy_impurity_left += self.sum_left[k] * self.sum_left[k]
#             proxy_impurity_right += self.sum_right[k] * self.sum_right[k]
#
#         return (proxy_impurity_left / self.weighted_n_left +
#                 proxy_impurity_right / self.weighted_n_right)
#
#     cdef void children_impurity(self, double* impurity_left,
#                                 double* impurity_right) nogil:
#         """Evaluate the impurity in children nodes.
#
#         i.e. the impurity of the left child (samples[start:pos]) and the
#         impurity the right child (samples[pos:end]).
#         """
#         cdef DOUBLE_t* sample_weight = self.sample_weight
#         cdef SIZE_t* samples = self.samples
#         cdef SIZE_t pos = self.pos
#         cdef SIZE_t start = self.start
#
#         cdef DOUBLE_t y_ik
#
#         cdef double sq_sum_left = 0.0
#         cdef double sq_sum_right
#
#         cdef SIZE_t i
#         cdef SIZE_t p
#         cdef SIZE_t k
#         cdef DOUBLE_t w = 1.0
#
#         for p in range(start, pos):
#             i = samples[p]
#
#             if sample_weight != NULL:
#                 w = sample_weight[i]
#
#             for k in range(self.n_outputs):
#                 y_ik = self.y[i, k]
#                 sq_sum_left += w * y_ik * y_ik
#
#         sq_sum_right = self.sq_sum_total - sq_sum_left
#
#         impurity_left[0] = sq_sum_left / self.weighted_n_left
#         impurity_right[0] = sq_sum_right / self.weighted_n_right
#
#         for k in range(self.n_outputs):
#             impurity_left[0] -= (self.sum_left[k] / self.weighted_n_left) ** 2.0
#             impurity_right[0] -= (self.sum_right[k] / self.weighted_n_right) ** 2.0
#
#         impurity_left[0] /= self.n_outputs
#         impurity_right[0] /= self.n_outputs
