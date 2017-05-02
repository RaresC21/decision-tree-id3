import numpy as np
cimport numpy as np

ctypedef np.npy_intp SIZE_t
ctypedef np.npy_int32 INT32_t
ctypedef np.npy_float32 DTYPE_t


cdef class TreeBuilder:
    cdef Splitter splitter
    cdef SIZE_t value_encoded
    cdef str value_decoded
    cdef SIZE_t size

cdef class Splitter:
