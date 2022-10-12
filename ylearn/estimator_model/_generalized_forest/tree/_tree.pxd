# This is a fork from scikit-learn

import numpy as np
cimport numpy as np

from ylearn.sklearn_ex.cloned.tree._tree cimport Tree
from ylearn.sklearn_ex.cloned.tree._tree cimport BestFirstTreeBuilder
from ._splitter cimport GrfTreeBestSplitter

cdef class GrfTreeBestFirstBuilder(BestFirstTreeBuilder):
    cpdef build_ex(self, Tree tree, object X, np.ndarray y, np.ndarray treatment,
                   np.ndarray sample_weight= *)

    cdef _init_splitter_ex(self, GrfTreeBestSplitter splitter, object X, np.ndarray y, np.ndarray treatment,
                           np.ndarray sample_weight= *)
