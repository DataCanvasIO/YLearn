
from ylearn.sklearn_ex.cloned.tree._splitter cimport BestSplitter

from ylearn.sklearn_ex.cloned.tree._tree cimport DOUBLE_t         # Type of y, sample_weight
from ylearn.sklearn_ex.cloned.tree._tree cimport SIZE_t           # Type for indices and counters

cdef class GrfTreeBestSplitter(BestSplitter):
    cdef const DOUBLE_t[:, ::1] treatment

    # new Methods
    cdef int init_ex(self, object X, const DOUBLE_t[:, ::1] y, const DOUBLE_t[:, ::1] treatment,
                  DOUBLE_t* sample_weight) except -1

    # overridden
    cdef int node_reset(self, SIZE_t start, SIZE_t end,
                        double* weighted_n_node_samples) nogil except -1
