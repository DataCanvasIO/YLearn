
import numpy as np
cimport numpy as np
from libc.stdio cimport printf


np.import_array()

from ylearn.sklearn_ex.cloned.tree._splitter cimport BestSplitter

from ylearn.sklearn_ex.cloned.tree._tree cimport DOUBLE_t
from ylearn.sklearn_ex.cloned.tree._tree cimport SIZE_t

from ._criterion cimport CriterionEx

cdef class GrfTreeBestSplitter(BestSplitter):
    cdef int init_ex(self, object X, const DOUBLE_t[:, ::1] y, const DOUBLE_t[:, ::1] treatment,
                  DOUBLE_t* sample_weight) except -1:
        self.init(X,y,sample_weight)
        self.treatment = treatment

    cdef int node_reset(self, SIZE_t start, SIZE_t end,
                        double* weighted_n_node_samples) nogil except -1:
        self.start = start
        self.end = end

        (<CriterionEx>self.criterion).init_ex(self.y,
                            self.treatment,
                            self.sample_weight,
                            self.weighted_n_samples,
                            self.samples,
                            start,
                            end)

        weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples
        return 0
