
import numpy as np
cimport numpy as np

from ylearn.sklearn_ex.cloned.tree._tree cimport DOUBLE_t
from ylearn.sklearn_ex.cloned.tree._tree cimport SIZE_t

cdef extern from "_util_lib.cpp":
    int eigen_solve(int m, int n, int r, double* A, double *B, double *X) nogil except -1
    int eigen_solve_r(int m, int n, int r, double* A, double *B, double *X) nogil except -1
    int eigen_pinv(int m, int n, const double *A, double *AI) nogil except -1
    int eigen_pinv_r(int m, int n, const double *A, double *AI) nogil except -1

    int init_criterion( #/* input */
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


cpdef solve(double[:, ::1] A, double[:, ::1] B):
    assert A.shape[0] == B.shape[0]

    cdef int m=A.shape[0]
    cdef int n=A.shape[1]
    cdef int r=B.shape[1]
    cdef np.ndarray[DOUBLE_t, ndim=2] X = np.zeros([n, r], dtype=np.float64)

    cdef int code= eigen_solve_r(m,n,r,&A[0,0],&B[0,0],&X[0,0])
    if code==0:
        return X
    else:
        raise ValueError(f'Failed to solve, code={code}')

cpdef pinv(double[:, ::1] A):
    cdef int m=A.shape[0]
    cdef int n=A.shape[1]

    cdef np.ndarray[DOUBLE_t, ndim=2] AI = np.zeros([n,m], dtype=np.float64)
    cdef int code = eigen_pinv_r(m,n,&A[0,0], &AI[0,0])

    if code==0:
        return AI
    else:
        raise ValueError(f'Failed to pinv, code={code}')
