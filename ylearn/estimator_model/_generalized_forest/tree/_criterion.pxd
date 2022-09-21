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
    pass
