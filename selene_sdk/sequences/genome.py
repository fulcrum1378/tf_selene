import pkg_resources
import pyfaidx
import tabix

from functools import wraps
from .sequence import Sequence
from .sequence import sequence_to_encoding
from .sequence import encoding_to_sequence


def _not_blacklist_region(chrom, start, end, blacklist_tabix):
    if blacklist_tabix is not None:
        try:
            rows = blacklist_tabix.query(chrom, start, end)
            for _ in rows:
                return False
        except tabix.TabixError:
            pass
    return True


def _check_coords(len_chrs, chrom, start, end, pad=False, blacklist_tabix=None):
    return chrom in len_chrs and \
           start < len_chrs[chrom] and \
           start < end and \
           end > 0 and \
           (start >= 0 if not pad else True) and \
           (end <= len_chrs[chrom] if not pad else True) and \
           _not_blacklist_region(chrom, start, end, blacklist_tabix)


def _get_sequence_from_coords(len_chrs, genome_sequence, chrom, start, end, strand='+', pad=False,
                              blacklist_tabix=None):
    if not _check_coords(len_chrs,
                         chrom,
                         start,
                         end,
                         pad=pad,
                         blacklist_tabix=blacklist_tabix):
        return ""

    if strand != '+' and strand != '-' and strand != '.':
        raise ValueError("Strand must be one of '+', '-', or '.'. Input was {0}".format(strand))

    end_pad = 0
    start_pad = 0
    if end > len_chrs[chrom]:
        end_pad = end - len_chrs[chrom]
        end = len_chrs[chrom]
    if start < 0:
        start_pad = -1 * start
        start = 0
    return (Genome.UNK_BASE * start_pad +
            genome_sequence(chrom, start, end, strand) +
            Genome.UNK_BASE * end_pad)


class Genome(Sequence):
    BASES_ARR = ['A', 'C', 'G', 'T']
    BASE_TO_INDEX = {
        'A': 0, 'C': 1, 'G': 2, 'T': 3,
        'a': 0, 'c': 1, 'g': 2, 't': 3,
    }
    INDEX_TO_BASE = {
        0: 'A', 1: 'C', 2: 'G', 3: 'T'
    }
    COMPLEMENTARY_BASE_DICT = {
        'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N',
        'a': 'T', 'c': 'G', 'g': 'C', 't': 'A', 'n': 'N'
    }
    UNK_BASE = "N"

    def __init__(self, input_path, blacklist_regions=None, bases_order=None, init_unpicklable=False):
        self.input_path = input_path
        self.blacklist_regions = blacklist_regions
        self._initialized = False

        if bases_order is not None:
            bases = [str.upper(b) for b in bases_order]
            self.BASES_ARR = bases
            lc_bases = [str.lower(b) for b in bases]
            self.BASE_TO_INDEX = {
                **{b: ix for (ix, b) in enumerate(bases)},
                **{b: ix for (ix, b) in enumerate(lc_bases)}}
            self.INDEX_TO_BASE = {ix: b for (ix, b) in enumerate(bases)}
            self.update_bases_order(bases)

        if init_unpicklable:
            self._unpicklable_init()

    @classmethod
    def update_bases_order(cls, bases):
        cls.BASES_ARR = bases
        lc_bases = [str.lower(b) for b in bases]
        cls.BASE_TO_INDEX = {
            **{b: ix for (ix, b) in enumerate(bases)},
            **{b: ix for (ix, b) in enumerate(lc_bases)}}
        cls.INDEX_TO_BASE = {ix: b for (ix, b) in enumerate(bases)}

    def _unpicklable_init(self):
        if not self._initialized:
            self.genome = pyfaidx.Fasta(self.input_path)
            self.chrs = sorted(self.genome.keys())
            self.len_chrs = self._get_len_chrs()
            self._blacklist_tabix = None

            if self.blacklist_regions == "hg19":
                self._blacklist_tabix = tabix.open(
                    pkg_resources.resource_filename(
                        "selene_sdk",
                        "sequences/data/hg19_blacklist_ENCFF001TDO.bed.gz"))
            elif self.blacklist_regions == "hg38":
                self._blacklist_tabix = tabix.open(
                    pkg_resources.resource_filename(
                        "selene_sdk",
                        "sequences/data/hg38.blacklist.bed.gz"))
            elif self.blacklist_regions is not None:  # user-specified file
                self._blacklist_tabix = tabix.open(
                    self.blacklist_regions)
            self._initialized = True

    def init(self):
        # delay initialization to allow  multiprocessing
        @wraps(self)
        def dfunc(self, *args, **kwargs):
            self._unpicklable_init()
            return self(self, *args, **kwargs)

        return dfunc

    @init
    def get_chrs(self):
        return self.chrs

    @init
    def get_chr_lens(self):
        return [(k, self.len_chrs[k]) for k in self.get_chrs()]

    def _get_len_chrs(self):
        len_chrs = {}
        for chrom in self.chrs:
            len_chrs[chrom] = len(self.genome[chrom])
        return len_chrs

    def _genome_sequence(self, chrom, start, end, strand='+'):
        if strand == '+' or strand == '.':
            return self.genome[chrom][start:end].seq
        else:
            return self.genome[chrom][start:end].reverse.complement.seq

    @init
    def coords_in_bounds(self, chrom, start, end):
        return _check_coords(self.len_chrs,
                             chrom,
                             start,
                             end,
                             blacklist_tabix=self._blacklist_tabix)

    @init
    def get_sequence_from_coords(self, chrom, start, end, strand='+', pad=False):
        return _get_sequence_from_coords(self.len_chrs,
                                         self._genome_sequence,
                                         chrom,
                                         start,
                                         end,
                                         strand=strand,
                                         pad=pad,
                                         blacklist_tabix=self._blacklist_tabix)

    @init
    def get_encoding_from_coords(self,
                                 chrom,
                                 start,
                                 end,
                                 strand='+',
                                 pad=False):
        sequence = self.get_sequence_from_coords(
            chrom, start, end, strand=strand, pad=pad)
        encoding = self.sequence_to_encoding(sequence)
        return encoding

    @init
    def get_encoding_from_coords_check_unk(self,
                                           chrom,
                                           start,
                                           end,
                                           strand='+',
                                           pad=False):
        sequence = self.get_sequence_from_coords(
            chrom, start, end, strand=strand, pad=pad)
        encoding = self.sequence_to_encoding(sequence)
        return encoding, self.UNK_BASE in sequence

    @classmethod
    def sequence_to_encoding(cls, sequence):
        return sequence_to_encoding(sequence, cls.BASE_TO_INDEX, cls.BASES_ARR)

    @classmethod
    def encoding_to_sequence(cls, encoding):
        return encoding_to_sequence(encoding, cls.BASES_ARR, cls.UNK_BASE)
