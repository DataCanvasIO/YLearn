
from ylearn.sklearn_ex.cloned.tree._criterion cimport RegressionCriterion

cdef class HonestCMSE(RegressionCriterion):
    cdef double yt_sq_sum_total
    cdef double y0_sq_sum_total

    cdef double[::1] yt_sum_total
    cdef double[::1] y0_sum_total

    cdef double[::1] yt_sum_left
    cdef double[::1] y0_sum_left
    cdef double[::1] yt_sum_right
    cdef double[::1] y0_sum_right

    cdef double nt_total
    cdef double n0_total

    cdef double nt_left
    cdef double n0_left

    cdef double nt_right
    cdef double n0_right