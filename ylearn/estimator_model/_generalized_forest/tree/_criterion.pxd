from ylearn.sklearn_ex.cloned.tree._criterion cimport Criterion

from ylearn.sklearn_ex.cloned.tree._tree cimport DOUBLE_t
from ylearn.sklearn_ex.cloned.tree._tree cimport SIZE_t

cdef class CriterionEx(Criterion):
    """
        Criterion with treatment
    """

    cdef const DOUBLE_t[:, ::1] treatment

    cdef int init_ex(self, const DOUBLE_t[:, ::1] y, const DOUBLE_t[:, ::1] treatment, DOUBLE_t* sample_weight, # fix
                  double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                  SIZE_t end) nogil except -1


cdef class GrfTreeCriterion(CriterionEx):
    cdef SIZE_t d_tr

    cdef double sq_sum_total

    cdef double[::1] sum_total   # The sum of w*y.
    cdef double[::1] sum_left    # Same as above, but for the left side of the split
    cdef double[::1] sum_right   # Same as above, but for the right side of the split

    cdef double[:, ::1] grad    
    cdef double[::1] sum_tr
    cdef double[::1] mean_tr
    cdef double[::1] mean_sum

    cdef double[::1] rho
    cdef double sum_rho
    cdef double sum_rho_left 
    cdef double sum_rho_right 
#
# cdef class TESTGrfTreeCriterion(CriterionEx):
#     cdef SIZE_t d_tr
#
#     cdef double sq_sum_total
#
#     cdef double[::1] sum_total   # The sum of w*y.
#     cdef double[::1] sum_left    # Same as above, but for the left side of the split
#     cdef double[::1] sum_right   # Same as above, but for the right side of the split
#
#     cdef double[:, ::1] grad
#     cdef double[::1] sum_tr
#     cdef double[::1] mean_tr
#     cdef double[::1] mean_sum
#
#     cdef double[::1] rho
#     cdef double sum_rho
#     cdef double sum_rho_left
#     cdef double sum_rho_right
#
# cdef class TestMSE(CriterionEx):
#     cdef SIZE_t d_tr
#
#     cdef double sq_sum_total
#
#     cdef double[::1] sum_total   # The sum of w*y.
#     cdef double[::1] sum_left    # Same as above, but for the left side of the split
#     cdef double[::1] sum_right   # Same as above, but for the right side of the split
