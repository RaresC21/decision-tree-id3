import numpy as np
cimport numpy as np
from .utils import unique


cdef class SplitRecord:
    LESS = 0
    GREATER = 1

    def __init__(self, calc_record, bag, SIZE_t value_encoded, value_decoded=None):
        self.calc_record = calc_record
        self.bag = bag
        self.value_encoded = value_encoded
        self.value_decoded = value_decoded
        self.size = len(bag) if bag is not None else 0


cdef class CalcRecord:
    NUM = 0
    NOM = 1

    cdef int init(self,
                 bint split_type,
                 DTYPE_t info,
                 SIZE_t feature_idx=-1,
                 str feature_name=None,
                 DTYPE_t entropy=-1,
                 DTYPE_t pivot=-1,
                 np.ndarray attribute_counts=None,
                 np.ndarray class_counts=None):
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

    cdef int init(self, object X, np.ndarray[SIZE_t, ndim=2] y, object is_numerical, object encoders, bint gain_ratio=False):
        self.X = X
        self.y = y
        self.is_numerical = is_numerical
        self.encoders = encoders
        self.gain_ratio = gain_ratio

    def _entropy(self, y, return_class_counts=False):
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
        cdef int n = y.shape[0]
        cdef float res
        cdef np.ndarray classes, counts, p, class_counts

        if n <= 0:
            return 0
        classes, count = unique(y)
        p = np.true_divide(count, n)
        res = - np.sum(np.multiply(p, np.log2(p)))
        if return_class_counts:
            class_counts = np.vstack((classes, count)).T
            return res, class_counts
        else:
            return res

    cdef CalcRecord _info_nominal(self, np.ndarray x, np.ndarray y):
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
        cdef float info = 0
        cdef SIZE_t n = x.shape[0]
        cdef np.ndarray items, count
        cdef CalcRecord cr
        items, count = unique(x)
        for value, p in zip(items, count):
            info += p * self._entropy(y[x == value])
        cr =  CalcRecord(CalcRecord.NOM,
                         info * np.true_divide(1, n),
                         entropy=0,
                         attribute_counts=count)
        return cr

    def _info_numerical(self, x, y):
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
        n = x.size
        if np.max(x) == np.min(x):
            return CalcRecord(None,
                              self._entropy(y),
                              attribute_counts=np.array([n]))
        sorted_idx = np.argsort(x, kind='quicksort')
        sorted_y = np.take(y, sorted_idx, axis=0)
        sorted_x = np.take(x, sorted_idx, axis=0)
        min_info = float('inf')
        min_info_pivot = 0
        min_attribute_counts = np.empty(2)
        for i in range(1, n):
            if sorted_x[i - 1] != sorted_x[i]:
                tmp_info = i * self._entropy(sorted_y[0: i]) + \
                           (n - i) * self._entropy(sorted_y[i:])
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

    def _split_numerical(self, X_, examples_idx, calc_record):
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

    def  _is_better(self, calc_record1, calc_record2):
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

    cdef CalcRecord calc(self, np.ndarray examples_idx, np.ndarray features_idx):
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
        cdef np.ndarray X_, y_, class_counts
        cdef CalcRecord calc_record, tmp_calc_record
        cdef float entropy

        X_ = self.X[np.ix_(examples_idx, features_idx)]
        y_ = self.y[examples_idx]
        calc_record = None
        entropy, class_counts = self._entropy(y_, True)
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

    def split(self, np.ndarray examples_idx, np.ndarray calc_record):
        cdef np.ndarray X_
        X_ = self.X[np.ix_(examples_idx)]
        if self.is_numerical[calc_record.feature_idx]:
            return self._split_numerical(X_, examples_idx, calc_record)
        else:
            return self._split_nominal(X_, examples_idx, calc_record)
