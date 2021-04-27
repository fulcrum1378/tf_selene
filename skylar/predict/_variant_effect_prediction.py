import math

import numpy as np

from ._common import _truncate_sequence
from ._common import predict

VCF_REQUIRED_COLS = ["#CHROM", "POS", "ID", "REF", "ALT"]


def read_vcf_file(input_path,
                  strand_index=None,
                  require_strand=False,
                  output_NAs_to_file=None,
                  seq_context=None,
                  reference_sequence=None):
    variants = []
    na_rows = []
    check_chr = True
    for chrom in reference_sequence.get_chrs():
        if not chrom.startswith("chr"):
            check_chr = False
            break
    with open(input_path, 'r') as file_handle:
        lines = file_handle.readlines()
        index = 0
        for index, line in enumerate(lines):
            if '#' not in line:
                break
            if "#CHROM" in line:
                cols = line.strip().split('\t')
                if cols[:5] != VCF_REQUIRED_COLS:
                    raise ValueError(
                        "First 5 columns in file {0} were {1}. "
                        "Expected columns: {2}".format(
                            input_path, cols[:5], VCF_REQUIRED_COLS))
                index += 1
                break
        for line in lines[index:]:
            cols = line.strip().split('\t')
            if len(cols) < 5:
                na_rows.append(line)
                continue
            chrom = str(cols[0])
            if 'CHR' == chrom[:3]:
                chrom = chrom.replace('CHR', 'chr')
            elif "chr" not in chrom and check_chr is True:
                chrom = "chr" + chrom

            if chrom == "chrMT" and \
                    chrom not in reference_sequence.get_chrs():
                chrom = "chrM"
            elif chrom == "MT" and \
                    chrom not in reference_sequence.get_chrs():
                chrom = "M"

            pos = int(cols[1])
            name = cols[2]
            ref = cols[3]
            if ref == '-':
                ref = ""
            alt = cols[4]
            strand = '+'
            if strand_index is not None:
                if require_strand and cols[strand_index] == '.':
                    na_rows.append(line)
                    continue
                elif cols[strand_index] == '-':
                    strand = '-'

            if reference_sequence and seq_context:
                if isinstance(seq_context, int):
                    seq_context = (seq_context, seq_context)
                lhs_radius, rhs_radius = seq_context
                start = pos + len(ref) // 2 - lhs_radius
                end = pos + len(ref) // 2 + rhs_radius
                if not reference_sequence.coords_in_bounds(chrom, start, end):
                    na_rows.append(line)
                    continue
            alt = alt.replace('.', ',')  # consider '.' a valid delimiter
            for a in alt.split(','):
                variants.append((chrom, pos, name, ref, a, strand))

    if reference_sequence and seq_context and output_NAs_to_file:
        with open(output_NAs_to_file, 'w') as file_handle:
            for na_row in na_rows:
                file_handle.write(na_row)
    return variants


def _get_ref_idxs(seq_len, ref_len):
    mid = seq_len // 2
    if seq_len % 2 == 0:
        mid -= 1
    start_pos = mid - ref_len // 2
    end_pos = start_pos + ref_len
    return (start_pos, end_pos)


def _process_alt(chrom,
                 pos,
                 ref,
                 alt,
                 start,
                 end,
                 wt_sequence,
                 reference_sequence):
    if alt == '*' or alt == '-':  # indicates a deletion
        alt = ''
    ref_len = len(ref)
    alt_len = len(alt)
    if alt_len > len(wt_sequence):
        sequence = _truncate_sequence(alt, len(wt_sequence))
        return reference_sequence.sequence_to_encoding(
            sequence)

    alt_encoding = reference_sequence.sequence_to_encoding(alt)
    if ref_len == alt_len:  # substitution
        start_pos, end_pos = _get_ref_idxs(len(wt_sequence), ref_len)
        sequence = np.vstack([wt_sequence[:start_pos, :],
                              alt_encoding,
                              wt_sequence[end_pos:, :]])
        return sequence
    elif alt_len > ref_len:  # insertion
        start_pos, end_pos = _get_ref_idxs(len(wt_sequence), ref_len)
        sequence = np.vstack([wt_sequence[:start_pos, :],
                              alt_encoding,
                              wt_sequence[end_pos:, :]])
        trunc_s = (len(sequence) - wt_sequence.shape[0]) // 2
        trunc_e = trunc_s + wt_sequence.shape[0]
        sequence = sequence[trunc_s:trunc_e, :]
        return sequence
    else:  # deletion
        lhs = reference_sequence.get_sequence_from_coords(
            chrom,
            start - ref_len // 2 + alt_len // 2,
            pos + 1,
            pad=True)
        rhs = reference_sequence.get_sequence_from_coords(
            chrom,
            pos + 1 + ref_len,
            end + math.ceil(ref_len / 2.) - math.ceil(alt_len / 2.),
            pad=True)
        sequence = lhs + alt + rhs
        return reference_sequence.sequence_to_encoding(
            sequence)


def _handle_standard_ref(ref_encoding,
                         seq_encoding,
                         seq_length,
                         reference_sequence):
    ref_len = ref_encoding.shape[0]

    start_pos, end_pos = _get_ref_idxs(seq_length, ref_len)

    sequence_encoding_at_ref = seq_encoding[
                               start_pos:start_pos + ref_len, :]
    references_match = np.array_equal(
        sequence_encoding_at_ref, ref_encoding)

    sequence_at_ref = None
    if not references_match:
        sequence_at_ref = reference_sequence.encoding_to_sequence(
            sequence_encoding_at_ref)
        seq_encoding[start_pos:start_pos + ref_len, :] = \
            ref_encoding
    return references_match, seq_encoding, sequence_at_ref


def _handle_long_ref(ref_encoding,
                     seq_encoding,
                     start_radius,
                     end_radius,
                     reference_sequence):
    ref_len = ref_encoding.shape[0]
    sequence_encoding_at_ref = seq_encoding
    ref_start = ref_len // 2 - start_radius - 1
    ref_end = ref_len // 2 + end_radius - 1
    ref_encoding = ref_encoding[ref_start:ref_end]
    references_match = np.array_equal(
        sequence_encoding_at_ref, ref_encoding)

    sequence_at_ref = None
    if not references_match:
        sequence_at_ref = reference_sequence.encoding_to_sequence(
            sequence_encoding_at_ref)
        seq_encoding = ref_encoding
    return references_match, seq_encoding, sequence_at_ref


def _handle_ref_alt_predictions(model,
                                batch_ref_seqs,
                                batch_alt_seqs,
                                batch_ids,
                                reporters,
                                use_cuda=False):
    batch_ref_seqs = np.array(batch_ref_seqs)
    batch_alt_seqs = np.array(batch_alt_seqs)
    ref_outputs = predict(model, batch_ref_seqs, use_cuda=use_cuda)
    alt_outputs = predict(model, batch_alt_seqs, use_cuda=use_cuda)
    for r in reporters:
        if r.needs_base_pred:
            r.handle_batch_predictions(alt_outputs, batch_ids, ref_outputs)
        else:
            r.handle_batch_predictions(alt_outputs, batch_ids)
