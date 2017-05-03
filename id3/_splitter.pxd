import numpy as np
cimport numpy as np

ctypedef np.npy_intp SIZE_t
ctypedef np.npy_int32 INT32_t
ctypedef np.npy_float32 DTYPE_t

cdef class CalcRecord:
    cdef public bint split_type
    cdef public DTYPE_t info
    cdef public SIZE_t feature_idx
    cdef public str feature_name
    cdef public DTYPE_t entropy
    cdef public DTYPE_t pivot
    cdef public np.ndarray class_counts
    cdef public np.ndarray attribute_counts

    cdef bint __lt__(self, CalcRecord other)

cdef class SplitRecord:
    cdef public CalcRecord calc_record
    cdef public np.ndarray bag
    cdef public SIZE_t value_encoded
    cdef public object value_decoded
    cdef public SIZE_t size

cdef class Splitter:
    cdef public np.ndarray X
    cdef public np.ndarray y
    cdef public object is_numerical
    cdef public object encoders
    cdef public bint gain_ratio

    cdef CalcRecord _info_nominal(self, np.ndarray x, np.ndarray y)
    cdef CalcRecord _info_numerical(self, np.ndarray x, np.ndarray y)
    #cdef SplitRecord _split_nominal(self, np.ndarray X_, np.ndarray examples_idx, CalcRecord calc_record)
    cdef list _split_numerical(self, np.ndarray X_, np.ndarray examples_idx, CalcRecord calc_record)
    #cdef object _entropy(self, np.ndarray y, bint return_class_counts=*)
    cpdef CalcRecord calc(self, np.ndarray examples_idx, np.ndarray features_idx)
    cdef bint _is_better(self, CalcRecord calc_record1, CalcRecord calc_record2)
    #cpdef CalcRecord split(self, np.ndarray examples_idx, CalcRecord calc_record)
