from abc import ABCMeta
from abc import abstractmethod

import numpy as np

from ._sequence import _fast_sequence_to_encoding


def sequence_to_encoding(sequence, base_to_index, bases_arr):
    return _fast_sequence_to_encoding(sequence, base_to_index, len(bases_arr))


def _get_base_index(encoding_row):
    unk_val = 1 / len(encoding_row)
    for index, val in enumerate(encoding_row):
        if np.isclose(val, unk_val) is True:
            return -1
        elif val == 1:
            return index
    return -1


def encoding_to_sequence(encoding, bases_arr, unk_base):
    sequence = []
    for row in encoding:
        base_pos = _get_base_index(row)
        if base_pos == -1:
            sequence.append(unk_base)
        else:
            sequence.append(bases_arr[base_pos])
    return "".join(sequence)


def get_reverse_encoding(encoding,
                         bases_arr,
                         base_to_index,
                         complementary_base_dict):
    reverse_encoding = np.zeros(encoding.shape)
    for index, row in enumerate(encoding):
        base_pos = _get_base_index(row)
        if base_pos == -1:
            reverse_encoding[index, :] = 1 / len(bases_arr)
        else:
            base = complementary_base_dict[bases_arr[base_pos]]
            complem_base_pos = base_to_index[base]
            rev_index = encoding.shape[0] - row - 1
            reverse_encoding[rev_index, complem_base_pos] = 1
    return reverse_encoding


def reverse_complement_sequence(sequence, complementary_base_dict):
    rev_comp_bases = [complementary_base_dict[b] for b in
                      sequence[::-1]]
    return ''.join(rev_comp_bases)


class Sequence(metaclass=ABCMeta):
    @property
    @abstractmethod
    def BASE_TO_INDEX(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def INDEX_TO_BASE(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def BASES_ARR(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def UNK_BASE(self):
        raise NotImplementedError()

    @abstractmethod
    def coords_in_bounds(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_sequence_from_coords(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_encoding_from_coords(self, *args, **kwargs):
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def sequence_to_encoding(cls, sequence):
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def encoding_to_sequence(cls, encoding):
        raise NotImplementedError()
