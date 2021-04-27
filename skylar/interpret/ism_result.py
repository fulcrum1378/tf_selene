import numpy as np
import pandas as pd

from ..sequences import Genome


class ISMResult(object):
    def __init__(self, data_frame, sequence_type=Genome):
        # Construct the reference sequence.
        alpha = set(sequence_type.BASES_ARR)
        ref_seq = [""] * (int(data_frame["pos"].iloc[-1]) + 1)
        seen = set()
        for row_idx, row in data_frame.iterrows():
            # Skip the reference value
            if not (row_idx == 0 and row["alt"] == "NA" and
                    row["ref"] == "NA"):
                cur_ref = row["ref"]
                if cur_ref not in alpha and cur_ref != sequence_type.UNK_BASE:
                    raise ValueError(
                        "Found character \'{0}\' from outside current alphabet"
                        " on row {1}.".format(cur_ref, row_idx))
                i = int(row["pos"])
                seen.add(i)
                if ref_seq[i] != "":
                    if ref_seq[i] != cur_ref:
                        raise Exception(
                            "Found 2 different letters for reference \'{0}\'"
                            " and \'{1}\' on row {2}.".format(ref_seq[i],
                                                              cur_ref,
                                                              row_idx))
                else:
                    ref_seq[i] = cur_ref
        if len(seen) != len(ref_seq):
            raise Exception(
                "Expected characters for {0} positions, but only found {1} of "
                "them.".format(len(ref_seq), len(seen)))
        ref_seq = "".join(ref_seq)
        self._reference_sequence = ref_seq
        self._data_frame = data_frame
        self._sequence_type = sequence_type

    @property
    def reference_sequence(self):
        return self._reference_sequence

    @property
    def sequence_type(self):
        return self._sequence_type

    def get_score_matrix_for(self, feature, reference_mask=None, dtype=np.float64):
        if reference_mask is not None:
            reference_mask = dtype(reference_mask)
        ret = self._sequence_type.sequence_to_encoding(
            self._reference_sequence).astype(dtype=dtype)
        ret[ret < 0] = 0.  # Set N's to zero to avoid spurious masking.
        alpha = set(self._sequence_type.BASES_ARR)
        for row_idx, row in self._data_frame.iterrows():
            # Extract reference value in first row.
            if row_idx == 0:
                if row["alt"] == "NA" and row["ref"] == "NA":
                    if reference_mask is None:
                        reference_mask = dtype(row[feature])
                    ret *= reference_mask
                    continue
                else:
                    if reference_mask is None:
                        reference_mask = 0.
                    ret *= reference_mask
            base = row["alt"]
            i = int(row["pos"])
            if base not in alpha:
                if base != self._sequence_type.UNK_BASE:
                    raise ValueError(
                        "Found character \'{0}\' from outside current alphabet"
                        " on row {1}.".format(base, row_idx))
            else:
                ret[i, self._sequence_type.BASE_TO_INDEX[base]] = dtype(
                    row[feature])
        return ret

    @staticmethod
    def from_file(input_path, sequence_type=Genome):
        return ISMResult(pd.read_csv(input_path, sep="\t", header=0,
                                     dtype=str, na_values=None,
                                     keep_default_na=False),
                         sequence_type=sequence_type)
