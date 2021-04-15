import types
from typing import List

import tabix
import numpy as np

from functools import wraps
from .target import Target
from ._genomic_features import _fast_get_feature_data


def _any_positive_rows(rows, start, end, thresholds):
    if rows is None:
        return False
    for row in rows:  # features within [start, end)
        is_positive = _is_positive_row(
            start, end, int(row[1]), int(row[2]), thresholds[row[3]])
        if is_positive:
            return True
    return False


def _is_positive_row(start, end,
                     feature_start, feature_end,
                     threshold):
    overlap_start = max(feature_start, start)
    overlap_end = min(feature_end, end)
    min_overlap_needed = int(
        (end - start) * threshold - 1)
    if min_overlap_needed < 0:
        min_overlap_needed = 0
    if overlap_end - overlap_start > min_overlap_needed:
        return True
    else:
        return False


def _get_feature_data(chrom, start, end,
                      thresholds, feature_index_dict, get_feature_rows):
    rows = get_feature_rows(chrom, start, end)
    return _fast_get_feature_data(
        start, end, thresholds, feature_index_dict, rows)


def _define_feature_thresholds(feature_thresholds, features):
    feature_thresholds_dict = {}
    feature_thresholds_vec = np.zeros(len(features))
    if (isinstance(feature_thresholds, float) or isinstance(feature_thresholds, int)) \
            and 1 >= feature_thresholds > 0:
        feature_thresholds_dict = dict.fromkeys(features, feature_thresholds)
        feature_thresholds_vec += feature_thresholds
    elif isinstance(feature_thresholds, dict):
        # assign the default value to everything first
        feature_thresholds_dict = dict.fromkeys(
            features, feature_thresholds["default"])
        feature_thresholds_vec += feature_thresholds["default"]
        for i, f in enumerate(features):
            if f in feature_thresholds:
                feature_thresholds_dict[f] = feature_thresholds[f]
                feature_thresholds_vec[i] = feature_thresholds[f]
    # this branch will not be accessed if you use a config.yml file to
    # specify input parameters
    elif isinstance(feature_thresholds, types.FunctionType):
        for i, f in enumerate(features):
            threshold = feature_thresholds(f)
            feature_thresholds_dict[f] = threshold
            feature_thresholds_vec[i] = threshold
    feature_thresholds_vec = feature_thresholds_vec.astype(np.float32)
    return feature_thresholds_dict, feature_thresholds_vec


class GenomicFeatures(Target):
    def __init__(self,
                 input_path: str,
                 features: List[str],
                 feature_thresholds=None,
                 init_unpicklable: bool = False):
        self.input_path = input_path
        self.n_features = len(features)

        self.feature_index_dict = dict(
            [(feat, index) for index, feat in enumerate(features)])

        self.index_feature_dict = dict(list(enumerate(features)))

        if feature_thresholds is None:
            self.feature_thresholds = None
            self._feature_thresholds_vec = None
        else:
            self.feature_thresholds, self._feature_thresholds_vec = \
                _define_feature_thresholds(feature_thresholds, features)
        self._initialized = False

        if init_unpicklable:
            self._unpicklable_init()

    def _unpicklable_init(self):
        if not self._initialized:
            self.data = tabix.open(self.input_path)
            self._initialized = True

    def init(self):
        # delay initialization to allow multiprocessing
        @wraps(self)
        def dfunc(self, *args, **kwargs):
            self._unpicklable_init()
            return self(self, *args, **kwargs)

        return dfunc

    def _query_tabix(self, chrom: str, start: int, end: int):
        try:
            return self.data.query(chrom, start, end)
        except tabix.TabixError:
            return None

    @init
    def is_positive(self, chrom: str, start: int, end: int) -> bool:
        rows = self._query_tabix(chrom, start, end)
        return _any_positive_rows(rows, start, end, self.feature_thresholds)

    @init
    def get_feature_data(self, chrom: str, start: int, end: int) -> np.ndarray:
        if self._feature_thresholds_vec is None:
            features = np.zeros(self.n_features)
            rows = self._query_tabix(chrom, start, end)
            if not rows:
                return features
            for r in rows:
                feature = r[3]
                ix = self.feature_index_dict[feature]
                features[ix] = 1
            return features
        return _get_feature_data(
            chrom, start, end, self._feature_thresholds_vec,
            self.feature_index_dict, self._query_tabix)
