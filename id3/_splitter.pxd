import numpy as np
cimport numpy as np

ctypedef np.npy_intp SIZE_t
ctypedef np.npy_int32 INT32_t
ctypedef np.npy_float32 DTYPE_t

cdef class CalcRecord:
    cdef bint split_type
    cdef DTYPE_t info
    cdef SIZE_t feature_idx
    cdef str feature_name
    cdef DTYPE_t entropy
    cdef DTYPE_t pivot
    cdef np.ndarray class_counts
    cdef np.ndarray attribute_counts

    cdef int init(self,
                 bint split_type,
                 DTYPE_t info,
                 SIZE_t feature_idx=*,
                 str feature_name=*,
                 DTYPE_t entropy=*,
                 DTYPE_t pivot=*,
                 np.ndarray attribute_counts=*,
                 np.ndarray class_counts=*)

    cdef bint __lt__(self, CalcRecord other)

cdef class SplitRecord:
    cdef CalcRecord calc_record
    cdef np.ndarray bag
    cdef SIZE_t value_encoded
    cdef str value_decoded
    cdef SIZE_t size

cdef class Splitter:

