from ylearn.sklearn_ex.cloned.tree._criterion cimport Criterion

cdef class GrfTreeCriterion(Criterion):
    cdef double sq_sum_total

    cdef double[::1] sum_total   # The sum of w*y.
    cdef double[::1] sum_left    # Same as above, but for the left side of the split
    cdef double[::1] sum_right   # Same as above, but for the right side of the split
    
    cdef double[::1] sum_tr
    cdef double[::1] mean_tr
    cdef double[::1] mean_sum

    cdef double[::1] sum_rho_total
    cdef double[::1] sum_rho_left 
    cdef double[::1] sum_rho_right 
    cdef double[::1] grad