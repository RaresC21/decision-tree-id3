import numpy as np
cimport numpy as np
from .utils import unique
import cython

from libc.math cimport log2


cdef class SplitRecord:
    LESS = 0
    GREATER = 1

    def __init__(self, CalcRecord calc_record, np.ndarray bag, SIZE_t value_encoded, object value_decoded=None):
        self.calc_record = calc_record
        self.bag = bag
        self.value_encoded = value_encoded
        self.value_decoded = value_decoded
        self.size = len(bag) if bag is not None else 0


cdef class CalcRecord:
    NUM = 0
    NOM = 1

    def __init__(self,
                 split_type,
                 info,
                 feature_idx=-1,
                 feature_name=None,
                 entropy=-1,
                 pivot=-1,
                 attribute_counts=None,
                 class_counts=None):
        self.split_type = split_type
        self.info = info
        self.feature_idx = feature_idx
        self.feature_name = feature_name
        self.entropy = entropy
        self.pivot = pivot
        self.class_counts = class_counts
        self.attribute_counts = attribute_counts

    cdef bint __lt__(self, CalcRecord other):
        if not isinstance(other, CalcRecord):
            return True
        return self.info < other.info


cdef class Splitter:

    def __init__(self, X, y, is_numerical, encoders, gain_ratio=False):
        self.X = X
        self.y = y
        self.is_numerical = is_numerical
        self.encoders = encoders
        self.gain_ratio = gain_ratio

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef object _entropy_full(self, np.ndarray[SIZE_t, ndim=1] y):
        """ Entropy for the classes in the array y
        :math: \sum_{x \in X} p(x) \log_{2}(1/p(x)) :math: from
        https://en.wikipedia.org/wiki/ID3_algorithm

        Parameters
        ----------
        y : nparray of shape [n remaining attributes]
            containing the class names

        Returns
        -------
        : float
            information for remaining examples given feature
        """
        cdef SIZE_t n = y.shape[0]
        cdef SIZE_t i
        cdef SIZE_t j
        cdef np.ndarray[SIZE_t, ndim=1] classes
        cdef np.ndarray[SIZE_t, ndim=1] count
        cdef np.ndarray[SIZE_t, ndim=2] class_counts
        cdef DTYPE_t res = 0
        cdef DTYPE_t p

        if n <= 0:
            return 0
        classes, count = unique(y)
        for i in range(count.shape[0]):
            p = count[i] / <float> n
            res = res - (p * log2(p))
        class_counts = np.vstack((classes, count)).T
        return res, class_counts

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef float _entropy(self, SIZE_t[:] y):
        """ Entropy for the classes in the array y
        :math: \sum_{x \in X} p(x) \log_{2}(1/p(x)) :math: from
        https://en.wikipedia.org/wiki/ID3_algorithm

        Parameters
        ----------
        y : nparray of shape [n remaining attributes]
            containing the class names

        Returns
        -------
        : float
            information for remaining examples given feature
        """
        cdef SIZE_t n = y.shape[0]
        cdef SIZE_t i
        cdef SIZE_t j
        cdef np.ndarray[INT32_t, ndim=1] count = np.zeros(y.shape[0], dtype=np.int32)
        cdef DTYPE_t res = 0
        cdef DTYPE_t p = 0

        if n == 0:
            return 0

        for i in range(y.shape[0]):
            count[y[i]] += 1

        for j in range(y.shape[0]):
            if count[j] == 0:
                continue
            p = count[j] / <float> n
            res = res - (p * log2(p))
        return res

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    def _info_nominal(self, np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[SIZE_t, ndim=1] y):
        """ Info for nominal feature feature_values
        :math: p(a)H(a) :math: from
        https://en.wikipedia.org/wiki/ID3_algorithm

        Parameters
        ----------
        x : np.array of shape [n remaining examples]
            containing feature values
        y : np.array of shape [n remaining examples]
            containing relevent class

        Returns
        -------
        : float
            information for remaining examples given feature
        """
        cdef DTYPE_t info
        cdef SIZE_t n = x.shape[0]
        cdef np.ndarray[SIZE_t, ndim=1] items, count
        cdef CalcRecord cr
        cdef SIZE_t i

        items, count = unique(x)
        for i in range(count.shape[0]):
            info += count[i] * self._entropy(y[x == items[i]])
        cr =  CalcRecord(CalcRecord.NOM,
                         info * np.true_divide(1, n),
                         entropy=0,
                         attribute_counts=count)
        return cr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef CalcRecord _info_numerical(self,
                                    np.ndarray[DTYPE_t, ndim=1] x,
                                    np.ndarray[SIZE_t, ndim=1] y):
        """ Info for numerical feature feature_values
        sort values then find the best split value

        Parameters
        ----------
        x : np.array of shape [n remaining examples]
            containing feature values
        y : np.array of shape [n remaining examples]
            containing relevent class

        Returns
        -------
        : float
            information for remaining examples given feature
        : float
            pivot used set1 < pivot <= set2
        """
        cdef SIZE_t n = x.shape[0]
        cdef CalcRecord cr
        cdef np.ndarray[SIZE_t, ndim=1] sorted_idx
        cdef np.ndarray[SIZE_t, ndim=1] sorted_y
        cdef np.ndarray[DTYPE_t, ndim=1] sorted_x
        cdef DTYPE_t min_info, tmp_info
        cdef DTYPE_t min_info_pivot = 0
        cdef np.ndarray[SIZE_t, ndim=1] min_attribute_counts = np.empty(2, dtype=np.intp)
        cdef np.ndarray[SIZE_t, ndim=1] tmp
        cdef SIZE_t i
        cdef SIZE_t j
        cdef DTYPE_t x_max = 0
        cdef DTYPE_t x_min = 0

        
        for i in range(n):
            if x[i] > x_max:
                x_max = x[i]
            if x[i] < x_min:
                x_min = x[i]
        if x_max == x_min:
            tmp = np.array([n], dtype=np.intp)
            cr =  CalcRecord(None,
                             self._entropy(y),
                             attribute_counts=tmp)
            return cr
        sorted_idx = np.argsort(x, kind='quicksort')
        sorted_y = np.empty(sorted_idx.shape[0], dtype=np.intp)
        sorted_x = np.empty(sorted_idx.shape[0], dtype=np.float32)
        for j in range(sorted_idx.shape[0]):
            sorted_y[j] = y[sorted_idx[j]]
            sorted_x[j] = x[sorted_idx[j]]
        min_info = np.inf
        min_info_pivot = 0
        for i in range(1, n):
            if sorted_x[i - 1] != sorted_x[i]:
                tmp_info = (i * self._entropy(sorted_y[0: i]) +
                           (n - i) * self._entropy(sorted_y[i:]))
                if tmp_info < min_info:
                    min_attribute_counts[SplitRecord.LESS] = n - i
                    min_attribute_counts[SplitRecord.GREATER] = i
                    min_info = tmp_info
                    min_info_pivot = (sorted_x[i - 1] + sorted_x[i]) / 2.0
        return CalcRecord(CalcRecord.NUM,
                          min_info * np.true_divide(1, n),
                          pivot=min_info_pivot,
                          attribute_counts=min_attribute_counts)

    def _split_nominal(self, X_, examples_idx, calc_record):
        ft_idx = calc_record.feature_idx
        values = self.encoders[ft_idx].encoded_classes_
        classes = self.encoders[ft_idx].classes_
        split_records = [None] * len(values)
        for val, i in enumerate(values):
            split_records[i] = SplitRecord(calc_record,
                                           examples_idx[X_[:, ft_idx] == val],
                                           val,
                                           classes[i])
        return split_records

    cdef list _split_numerical(self,
                                 np.ndarray X_,
                                 np.ndarray examples_idx,
                                 CalcRecord calc_record):
        cdef SIZE_t idx
        cdef list split_records
        idx = calc_record.feature_idx
        split_records = [None] * 2
        split_records[0] = SplitRecord(calc_record,
                                       examples_idx[X_[:, idx]
                                                    <= calc_record.pivot],
                                       SplitRecord.LESS)
        split_records[1] = SplitRecord(calc_record,
                                       examples_idx[X_[:, idx]
                                                    > calc_record.pivot],
                                       SplitRecord.GREATER)
        return split_records

    def _intrinsic_value(self, calc_record):
        """ Calculates the gain ratio using CalcRecord
        :math: - \sum_{i} \fraq{|S_i|}{|S|}\log_2 (\fraq{|S_i|}{|S|}):math:

        Parameters
        ----------
        calc_record : CalcRecord

        Returns
        -------
        : float
        """
        counts = calc_record.attribute_counts
        s = np.true_divide(counts, np.sum(counts))
        return - np.sum(np.multiply(s, np.log2(s)))

    cdef bint _is_better(self,
                         CalcRecord calc_record1,
                         CalcRecord calc_record2):
        """Compairs CalcRecords

        Parameters
        ----------
        calc_record1 : CalcRecord
        calc_record2 : CalcRecord

        Returns
        -------
        : bool
            if calc_record1 > calc_record2
        """
        cdef DTYPE_t info_gain1, info_gain2

        if calc_record1 is None:
            return True
        if calc_record2 is None:
            return False
        if calc_record2.split_type is None:
            return False
        if self.gain_ratio:
            info_gain1 = np.true_divide(calc_record1.entropy
                                        + calc_record1.info,
                                        self._intrinsic_value(calc_record1))
            info_gain2 = np.true_divide(calc_record2.entropy
                                        + calc_record2.info,
                                        self._intrinsic_value(calc_record2))
            return info_gain1 < info_gain2
        else:
            return calc_record1.info > calc_record2.info

    cpdef CalcRecord calc(self, np.ndarray examples_idx, np.ndarray features_idx):
        """ Calculates information regarding optimal split based on information
        gain

        Parameters
        ----------
        x : np.array of shape [n remaining examples]
            containing feature values
        y : np.array of shape [n remaining examples]
            containing relevent class

        Returns
        -------
        : float
            information for remaining examples given feature
        : float
            pivot used set1 < pivot <= set2
        """
        cdef np.ndarray X_, y_
        cdef CalcRecord calc_record, tmp_calc_record
        cdef DTYPE_t entropy
        cdef np.ndarray class_counts
        cdef SIZE_t idx
        cdef np.ndarray feature

        X_ = self.X[np.ix_(examples_idx, features_idx)]
        y_ = self.y[examples_idx]
        calc_record = None
        entropy, class_counts = self._entropy_full(y_)
        for idx, feature in enumerate(X_.T):
            tmp_calc_record = None
            if self.is_numerical[features_idx[idx]]:
                tmp_calc_record = self._info_numerical(feature, y_)
            else:
                tmp_calc_record = self._info_nominal(feature, y_)
            tmp_calc_record.entropy = entropy
            tmp_calc_record.class_counts = class_counts
            if self._is_better(calc_record, tmp_calc_record):
                calc_record = tmp_calc_record
                calc_record.feature_idx = features_idx[idx]
        return calc_record

    def split(self, examples_idx, calc_record):
        X_ = self.X[np.ix_(examples_idx)]
        if self.is_numerical[calc_record.feature_idx]:
            return self._split_numerical(X_, examples_idx, calc_record)
        else:
            return self._split_nominal(X_, examples_idx, calc_record)
