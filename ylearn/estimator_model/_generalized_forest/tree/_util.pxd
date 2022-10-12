from ylearn.sklearn_ex.cloned.tree._tree cimport SIZE_t

cdef int eigen_solve(int m, int n, int r, double* A, double *B, double *X) nogil except -1
# cdef int eigen_solve_r(int m, int n, int r, double* A, double *B, double *X) nogil except -1
cdef int eigen_pinv(int m, int n, const double *A, double *AI) nogil except -1
# cdef int eigen_pinv_r(int m, int n, const double *A, double *AI) nogil except -1

cdef int init_criterion( #/* input */
                    int d_y,int d_tr, int n_samples,
                    const double *y, const double *tr,
                    const double *sample_weight, const SIZE_t *samples,
                    int start, int end,
                    #/* output */
                    double *sum_total,  double *mean_sum,
                    double *sum_tr,double * mean_tr,
                    double *rho, double *grad,
                    double *weighted_n_node_samples,
                    double *sum_rho
) nogil except -1

#
cpdef solve(double[:, ::1] A, double[:, ::1] B)

cpdef pinv(double[:, ::1] A)
