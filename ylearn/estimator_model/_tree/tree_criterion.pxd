
from ylearn.sklearn_ex.cloned.tree._criterion cimport RegressionCriterion

cdef class HonestCMSE(RegressionCriterion):
    cdef double yt_sq_sum_total
    cdef double y0_sq_sum_total

    cdef double yt_sum_total
    cdef double y0_sum_total

    cdef double yt_sq_sum_left
    cdef double y0_sq_sum_left
    cdef double yt_sq_sum_right
    cdef double y0_sq_sum_right

    cdef double yt_sum_left
    cdef double y0_sum_left
    cdef double yt_sum_right
    cdef double y0_sum_right

    cdef int nt_total
    cdef int n0_total

    cdef int nt_left
    cdef int n0_left

    cdef int nt_right
    cdef int n0_right
