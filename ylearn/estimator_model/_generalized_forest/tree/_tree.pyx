import numpy as np
cimport numpy as np
from libc.stdio cimport printf

np.import_array()

from ylearn.sklearn_ex.cloned.tree._tree cimport Tree
from ylearn.sklearn_ex.cloned.tree._tree cimport BestFirstTreeBuilder

from ylearn.sklearn_ex.cloned.tree._tree cimport DOUBLE_t

from ._splitter cimport GrfTreeBestSplitter

cdef class GrfTreeBestFirstBuilder(BestFirstTreeBuilder):
    cpdef build_ex(self, Tree tree, object X, np.ndarray y, np.ndarray treatment,
                   np.ndarray sample_weight=None):
        """Build a decision tree from the training set (X, y, treatment)."""

        # check input
        X, y, sample_weight = self._check_input(X, y, sample_weight)

        # init splitter
        cdef GrfTreeBestSplitter splitter = <GrfTreeBestSplitter>self.splitter
        self._init_splitter_ex(splitter, X, y, treatment)
        # build tree
        self._build_tree(tree, splitter)

    cdef _init_splitter_ex(self, GrfTreeBestSplitter splitter, object X, np.ndarray y, np.ndarray treatment,
                           np.ndarray sample_weight= None):
        cdef DOUBLE_t*sample_weight_ptr = NULL
        if sample_weight is not None:
            sample_weight_ptr = <DOUBLE_t*> sample_weight.data

        # Recursive partition (without actual recursion)
        splitter.init_ex(X, y, treatment, sample_weight_ptr)
