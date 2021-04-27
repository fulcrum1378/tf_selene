import pyfaidx

from .sequence import Sequence
from .sequence import sequence_to_encoding
from .sequence import encoding_to_sequence


def _get_sequence_from_coords(len_prots, proteome_sequence, prot, start, end):
    if start > len_prots[prot] or (end > len_prots[prot] + 1) or start < 0:
        return ""
    return proteome_sequence(prot, start, end)


class Proteome(Sequence):
    BASES_ARR = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    BASE_TO_INDEX = {
        'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'E': 5, 'Q': 6,
        'G': 7, 'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13,
        'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
    }
    INDEX_TO_BASE = {0: 'A', 1: 'R', 2: 'N', 3: 'D', 4: 'C', 5: 'E', 6: 'Q',
                     7: 'G', 8: 'H', 9: 'I', 10: 'L', 11: 'K', 12: 'M',
                     13: 'F', 14: 'P', 15: 'S', 16: 'T', 17: 'W', 18: 'Y',
                     19: 'V'}
    UNK_BASE = "X"

    def __init__(self, input_path):
        self.proteome = pyfaidx.Fasta(input_path)
        self.prots = sorted(self.proteome.keys())
        self.len_prots = self._get_len_prots()

    def get_prots(self):
        return self.prots

    def get_prot_lens(self):
        return [(k, self.len_prots[k]) for k in self.prots]

    def _get_len_prots(self):
        len_prots = {}
        for prot in self.prots:
            len_prots[prot] = len(self.proteome[prot])
        return len_prots

    def _proteome_sequence(self, prot, start, end):
        return self.proteome[prot][start:end].seq

    def coords_in_bounds(self, prot, start, end):
        if (start > self.len_prots[prot] or end > (self.len_prots[prot] + 1)
                or start < 0):
            return False
        return True

    def get_sequence_from_coords(self, prot, start, end):
        return _get_sequence_from_coords(
            self.len_prots, self._proteome_sequence, prot, start, end)

    def get_encoding_from_coords(self, prot, start, end):
        sequence = self.get_sequence_from_coords(prot, start, end)
        encoding = self.sequence_to_encoding(sequence)
        return encoding

    @classmethod
    def sequence_to_encoding(cls, sequence):
        return sequence_to_encoding(sequence, cls.BASE_TO_INDEX, cls.BASES_ARR)

    @classmethod
    def encoding_to_sequence(cls, encoding):
        return encoding_to_sequence(encoding, cls.BASES_ARR, cls.UNK_BASE)
