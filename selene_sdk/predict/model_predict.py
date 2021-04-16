import math
import os
import warnings
from time import time
from typing import List, Tuple

import numpy as np
import pyfaidx
import tensorflow as tf
from torch import load

from ._common import _pad_sequence
from ._common import _truncate_sequence
from ._common import get_reverse_complement_encoding
from ._common import predict
from ._in_silico_mutagenesis import _ism_sample_id
from ._in_silico_mutagenesis import in_silico_mutagenesis_sequences
from ._in_silico_mutagenesis import mutate_sequence
from ._variant_effect_prediction import _handle_long_ref
from ._variant_effect_prediction import _handle_ref_alt_predictions
from ._variant_effect_prediction import _handle_standard_ref
from ._variant_effect_prediction import _process_alt
from ._variant_effect_prediction import read_vcf_file
from .predict_handlers import AbsDiffScoreHandler
from .predict_handlers import DiffScoreHandler
from .predict_handlers import LogitScoreHandler
from .predict_handlers import WritePredictionsHandler
from .predict_handlers import WriteRefAltHandler
from ..sequences import Genome
from ..utils import _is_lua_trained_model
from ..utils import load_model_from_state_dict

ISM_COLS = ["pos", "ref", "alt"]
VARIANTEFFECT_COLS = ["chrom", "pos", "name", "ref", "alt", "strand", "ref_match", "contains_unk"]


class AnalyzeSequences(object):
    def __init__(self,
                 model: tf.Module,
                 trained_model_path,  # str or List[str]
                 sequence_length: int,
                 features: List[str],
                 batch_size: int = 64,
                 use_cuda: bool = False,
                 # data_parallel: bool = False,
                 reference_sequence=Genome,
                 write_mem_limit: int = 1500):
        self.model = model

        if isinstance(trained_model_path, str):
            trained_model = load(
                trained_model_path,
                map_location=lambda storage, location: storage)

            load_model_from_state_dict(
                trained_model, self.model)
        elif hasattr(trained_model_path, '__len__'):
            state_dicts = []
            for mp in trained_model_path:
                state_dict = load(
                    mp, map_location=lambda storage, location: storage)
                state_dicts.append(state_dict)

            for (sd, sub_model) in zip(state_dicts, self.model.sub_models):
                load_model_from_state_dict(sd, sub_model)
        else:
            raise ValueError(
                '`trained_model_path` should be a str or list of strs '
                'specifying the full paths to model weights files, but was '
                'type {0}.'.format(type(trained_model_path)))

        self.model.eval()

        self.use_cuda = use_cuda
        if self.use_cuda: self.model.cuda()

        self.sequence_length = sequence_length

        self._start_radius = sequence_length // 2
        self._end_radius = self._start_radius
        if sequence_length % 2 != 0:
            self._start_radius += 1

        self.batch_size = batch_size
        self.features = features
        self.reference_sequence = reference_sequence
        if not self.reference_sequence._initialized:
            self.reference_sequence._unpicklable_init()
        if type(self.reference_sequence) == Genome and \
                _is_lua_trained_model(model):
            Genome.update_bases_order(['A', 'G', 'C', 'T'])
        else:  # even if not using Genome, I guess we can update?
            Genome.update_bases_order(['A', 'C', 'G', 'T'])
        self._write_mem_limit = write_mem_limit

    def _initialize_reporters(self,
                              save_data,
                              output_path_prefix,
                              output_format,
                              colnames_for_ids,
                              output_size=None,
                              mode="ism"):
        save_data = set(save_data) & {"diffs", "abs_diffs", "logits", "predictions"}
        save_data = sorted(list(save_data))
        if len(save_data) == 0:
            raise ValueError("'save_data' parameter must be a list that "
                             "contains one of ['diffs', 'abs_diffs', "
                             "'logits', 'predictions'].")
        reporters = []
        constructor_args = [self.features,
                            colnames_for_ids,
                            output_path_prefix,
                            output_format,
                            output_size,
                            self._write_mem_limit // len(save_data)]
        for i, s in enumerate(save_data):
            write_labels = False
            if i == 0:
                write_labels = True
            if "diffs" == s:
                reporters.append(DiffScoreHandler(
                    *constructor_args, write_labels=write_labels))
            elif "abs_diffs" == s:
                reporters.append(AbsDiffScoreHandler(
                    *constructor_args, write_labels=write_labels))
            elif "logits" == s:
                reporters.append(LogitScoreHandler(
                    *constructor_args, write_labels=write_labels))
            elif "predictions" == s and mode != "varianteffect":
                reporters.append(WritePredictionsHandler(
                    *constructor_args, write_labels=write_labels))
            elif "predictions" == s and mode == "varianteffect":
                reporters.append(WriteRefAltHandler(
                    *constructor_args, write_labels=write_labels))
        return reporters

    def _get_sequences_from_bed_file(self,
                                     input_path,
                                     strand_index=None,
                                     output_NAs_to_file=None,
                                     reference_sequence=None) -> Tuple:
        sequences = []
        labels = []
        na_rows = []
        check_chr = True
        for chrom in reference_sequence.get_chrs():
            if not chrom.startswith("chr"):
                check_chr = False
                break
        with open(input_path, 'r') as read_handle:
            for i, line in enumerate(read_handle):
                cols = line.strip().split('\t')
                if len(cols) < 3:
                    na_rows.append(line)
                    continue
                chrom = cols[0]
                start = cols[1]
                end = cols[2]
                strand = '.'
                if isinstance(strand_index, int) and len(cols) > strand_index:
                    strand = cols[strand_index]
                if 'chr' not in chrom and check_chr is True:
                    chrom = "chr{0}".format(chrom)
                if not str.isdigit(start) or not str.isdigit(end) \
                        or chrom not in self.reference_sequence.genome:
                    na_rows.append(line)
                    continue
                start, end = int(start), int(end)
                mid_pos = start + ((end - start) // 2)
                seq_start = mid_pos - self._start_radius
                seq_end = mid_pos + self._end_radius
                if reference_sequence:
                    if not reference_sequence.coords_in_bounds(chrom, seq_start, seq_end):
                        na_rows.append(line)
                        continue
                sequences.append((chrom, seq_start, seq_end, strand))
                labels.append((i, chrom, start, end, strand))

        if reference_sequence and output_NAs_to_file:
            with open(output_NAs_to_file, 'w') as file_handle:
                for na_row in na_rows:
                    file_handle.write(na_row)

        return sequences, labels

    def get_predictions_for_bed_file(self,
                                     input_path: str,
                                     output_dir: str,
                                     output_format: str = "tsv",
                                     strand_index: int = None) -> None:
        _, filename = os.path.split(input_path)
        output_prefix = '.'.join(filename.split('.')[:-1])

        seq_coords, labels = self._get_sequences_from_bed_file(
            input_path,
            strand_index=strand_index,
            output_NAs_to_file="{0}.NA".format(os.path.join(output_dir, output_prefix)),
            reference_sequence=self.reference_sequence)

        reporter = self._initialize_reporters(
            ["predictions"],
            os.path.join(output_dir, output_prefix),
            output_format,
            ["index", "chrom", "start", "end", "strand", "contains_unk"],
            output_size=len(labels),
            mode="prediction")[0]
        sequences = None
        batch_ids = []
        for i, (label, coords) in enumerate(zip(labels, seq_coords)):
            encoding, contains_unk = self.reference_sequence.get_encoding_from_coords_check_unk(
                *coords, pad=True)
            if sequences is None:
                sequences = np.zeros((self.batch_size, *encoding.shape))
            if i and i % self.batch_size == 0:
                preds = predict(self.model, sequences, use_cuda=self.use_cuda)
                reporter.handle_batch_predictions(preds, batch_ids)
                sequences = np.zeros((self.batch_size, *encoding.shape))
                batch_ids = []
            sequences[i % self.batch_size, :, :] = encoding
            batch_ids.append(label + (contains_unk,))
            if contains_unk:
                warnings.warn(("For region {0}, "
                               "reference sequence contains unknown "
                               "base(s). --will be marked `True` in the "
                               "`contains_unk` column of the .tsv or "
                               "row_labels .txt file.").format(label))

        sequences = sequences[:i % self.batch_size + 1, :, :]
        preds = predict(self.model, sequences, use_cuda=self.use_cuda)
        reporter.handle_batch_predictions(preds, batch_ids)
        reporter.write_to_file()

    def get_predictions_for_fasta_file(self,
                                       input_path: str,
                                       output_dir: str,
                                       output_format: str = "tsv") -> None:
        os.makedirs(output_dir, exist_ok=True)

        _, filename = os.path.split(input_path)
        output_prefix = '.'.join(filename.split('.')[:-1])

        fasta_file = pyfaidx.Fasta(input_path)
        reporter = self._initialize_reporters(
            ["predictions"],
            os.path.join(output_dir, output_prefix),
            output_format,
            ["index", "name"],
            output_size=len(fasta_file.keys()),
            mode="prediction")[0]
        sequences = np.zeros((self.batch_size,
                              self.sequence_length,
                              len(self.reference_sequence.BASES_ARR)))
        batch_ids = []
        for i, fasta_record in enumerate(fasta_file):
            cur_sequence = self._pad_or_truncate_sequence(str(fasta_record))
            cur_sequence_encoding = self.reference_sequence.sequence_to_encoding(
                cur_sequence)

            if i and i > 0 and i % self.batch_size == 0:
                preds = predict(self.model, sequences, use_cuda=self.use_cuda)
                sequences = np.zeros(
                    (self.batch_size, *cur_sequence_encoding.shape))
                reporter.handle_batch_predictions(preds, batch_ids)
                batch_ids = []

            batch_ids.append([i, fasta_record.name])
            sequences[i % self.batch_size, :, :] = cur_sequence_encoding

        sequences = sequences[:i % self.batch_size + 1, :, :]
        preds = predict(self.model, sequences, use_cuda=self.use_cuda)
        reporter.handle_batch_predictions(preds, batch_ids)

        fasta_file.close()
        reporter.write_to_file()

    def get_predictions(self,
                        input_path: str,
                        output_dir: str = None,
                        output_format: str = "tsv",
                        strand_index: int = None) -> None:
        if output_dir is None:
            sequence = self._pad_or_truncate_sequence(input_path)
            seq_enc = self.reference_sequence.sequence_to_encoding(sequence)
            seq_enc = np.expand_dims(seq_enc, axis=0)  # add batch size of 1
            return predict(self.model, seq_enc, use_cuda=self.use_cuda)
        elif input_path.endswith('.fa') or input_path.endswith('.fasta'):
            self.get_predictions_for_fasta_file(
                input_path, output_dir, output_format=output_format)
        else:
            self.get_predictions_for_bed_file(
                input_path,
                output_dir,
                output_format=output_format,
                strand_index=strand_index)

        return None

    def in_silico_mutagenesis_predict(self,
                                      sequence,
                                      base_preds,
                                      mutations_list,
                                      reporters=[]):
        current_sequence_encoding = self.reference_sequence.sequence_to_encoding(
            sequence)
        for i in range(0, len(mutations_list), self.batch_size):
            start = i
            end = min(i + self.batch_size, len(mutations_list))

            mutated_sequences = np.zeros(
                (end - start, *current_sequence_encoding.shape))

            batch_ids = []
            for ix, mutation_info in enumerate(mutations_list[start:end]):
                mutated_seq = mutate_sequence(
                    current_sequence_encoding, mutation_info,
                    reference_sequence=self.reference_sequence)
                mutated_sequences[ix, :, :] = mutated_seq
                batch_ids.append(_ism_sample_id(sequence, mutation_info))
            outputs = predict(
                self.model, mutated_sequences, use_cuda=self.use_cuda)

            for r in reporters:
                if r.needs_base_pred:
                    r.handle_batch_predictions(outputs, batch_ids, base_preds)
                else:
                    r.handle_batch_predictions(outputs, batch_ids)

        for r in reporters:
            r.write_to_file()

    def in_silico_mutagenesis(self,
                              sequence,
                              save_data,
                              output_path_prefix="ism",
                              mutate_n_bases=1,
                              output_format="tsv",
                              start_position=0,
                              end_position=None):
        if end_position is None:
            end_position = self.sequence_length
        if start_position >= end_position:
            raise ValueError(("Starting positions must be less than the ending "
                              "positions. Found a starting position of {0} with "
                              "an ending position of {1}.").format(start_position,
                                                                   end_position))
        if start_position < 0:
            raise ValueError("Negative starting positions are not supported.")
        if end_position < 0:
            raise ValueError("Negative ending positions are not supported.")
        if start_position >= self.sequence_length:
            raise ValueError(("Starting positions must be less than the sequence length."
                              " Found a starting position of {0} with a sequence length "
                              "of {1}.").format(start_position, self.sequence_length))
        if end_position > self.sequence_length:
            raise ValueError(("Ending positions must be less than or equal to the sequence "
                              "length. Found an ending position of {0} with a sequence "
                              "length of {1}.").format(end_position, self.sequence_length))
        if (end_position - start_position) < mutate_n_bases:
            raise ValueError(("Fewer bases exist in the substring specified by the starting "
                              "and ending positions than need to be mutated. There are only "
                              "{0} currently, but {1} bases must be mutated at a "
                              "time").format(end_position - start_position, mutate_n_bases))

        path_dirs, _ = os.path.split(output_path_prefix)
        if path_dirs:
            os.makedirs(path_dirs, exist_ok=True)

        n = len(sequence)
        if n < self.sequence_length:  # Pad string length as necessary.
            diff = (self.sequence_length - n) / 2
            pad_l = int(np.floor(diff))
            pad_r = math.ceil(diff)
            sequence = ((self.reference_sequence.UNK_BASE * pad_l) +
                        sequence +
                        (self.reference_sequence.UNK_BASE * pad_r))
        elif n > self.sequence_length:  # Extract center substring of proper length.
            start = int((n - self.sequence_length) // 2)
            end = int(start + self.sequence_length)
            sequence = sequence[start:end]

        sequence = str.upper(sequence)
        mutated_sequences = in_silico_mutagenesis_sequences(
            sequence, mutate_n_bases=1,
            reference_sequence=self.reference_sequence,
            start_position=start_position,
            end_position=end_position)
        reporters = self._initialize_reporters(
            save_data,
            output_path_prefix,
            output_format,
            ISM_COLS,
            output_size=len(mutated_sequences))

        current_sequence_encoding = \
            self.reference_sequence.sequence_to_encoding(sequence)

        current_sequence_encoding = current_sequence_encoding.reshape(
            (1, *current_sequence_encoding.shape))
        base_preds = predict(
            self.model, current_sequence_encoding, use_cuda=self.use_cuda)

        if "predictions" in save_data and output_format == 'hdf5':
            ref_reporter = self._initialize_reporters(
                ["predictions"],
                "{0}_ref".format(output_path_prefix),
                output_format, ["name"], output_size=1)[0]
            ref_reporter.handle_batch_predictions(
                base_preds, [["input_sequence"]])
            ref_reporter.write_to_file()
        elif "predictions" in save_data and output_format == 'tsv':
            reporters[-1].handle_batch_predictions(
                base_preds, [["input_sequence", "NA", "NA"]])

        self.in_silico_mutagenesis_predict(
            sequence,
            base_preds,
            mutated_sequences,
            reporters=reporters)

    def in_silico_mutagenesis_from_file(self,
                                        input_path,
                                        save_data,
                                        output_dir,
                                        mutate_n_bases=1,
                                        use_sequence_name=True,
                                        output_format="tsv",
                                        start_position=0,
                                        end_position=None):
        if end_position is None:
            end_position = self.sequence_length
        if start_position >= end_position:
            raise ValueError(("Starting positions must be less than the ending "
                              "positions. Found a starting position of {0} with "
                              "an ending position of {1}.").format(start_position,
                                                                   end_position))
        if start_position < 0:
            raise ValueError("Negative starting positions are not supported.")
        if end_position < 0:
            raise ValueError("Negative ending positions are not supported.")
        if start_position >= self.sequence_length:
            raise ValueError(("Starting positions must be less than the sequence length."
                              " Found a starting position of {0} with a sequence length "
                              "of {1}.").format(start_position, self.sequence_length))
        if end_position > self.sequence_length:
            raise ValueError(("Ending positions must be less than or equal to the sequence "
                              "length. Found an ending position of {0} with a sequence "
                              "length of {1}.").format(end_position, self.sequence_length))
        if (end_position - start_position) < mutate_n_bases:
            raise ValueError(("Fewer bases exist in the substring specified by the starting "
                              "and ending positions than need to be mutated. There are only "
                              "{0} currently, but {1} bases must be mutated at a "
                              "time").format(end_position - start_position, mutate_n_bases))

        os.makedirs(output_dir, exist_ok=True)

        fasta_file = pyfaidx.Fasta(input_path)
        for i, fasta_record in enumerate(fasta_file):
            cur_sequence = self._pad_or_truncate_sequence(str.upper(str(fasta_record)))

            # Generate mut sequences and base preds.
            mutated_sequences = in_silico_mutagenesis_sequences(
                cur_sequence,
                mutate_n_bases=mutate_n_bases,
                reference_sequence=self.reference_sequence,
                start_position=start_position,
                end_position=end_position)
            cur_sequence_encoding = self.reference_sequence.sequence_to_encoding(
                cur_sequence)
            base_encoding = cur_sequence_encoding.reshape(
                1, *cur_sequence_encoding.shape)
            base_preds = predict(
                self.model, base_encoding, use_cuda=self.use_cuda)

            if use_sequence_name:
                file_prefix = os.path.join(
                    output_dir, fasta_record.name.replace(' ', '_'))
            else:
                file_prefix = os.path.join(
                    output_dir, str(i))
            # Write base to file, and make mut preds.
            reporters = self._initialize_reporters(
                save_data,
                file_prefix,
                output_format,
                ISM_COLS,
                output_size=len(mutated_sequences))

            if "predictions" in save_data and output_format == 'hdf5':
                ref_reporter = self._initialize_reporters(
                    ["predictions"],
                    "{0}_ref".format(file_prefix),
                    output_format, ["name"], output_size=1)[0]
                ref_reporter.handle_batch_predictions(
                    base_preds, [["input_sequence"]])
                ref_reporter.write_to_file()
            elif "predictions" in save_data and output_format == 'tsv':
                reporters[-1].handle_batch_predictions(
                    base_preds, [["input_sequence", "NA", "NA"]])

            self.in_silico_mutagenesis_predict(
                cur_sequence, base_preds, mutated_sequences,
                reporters=reporters)
        fasta_file.close()

    def variant_effect_prediction(self,
                                  vcf_file,
                                  save_data,
                                  output_dir=None,
                                  output_format="tsv",
                                  strand_index=None,
                                  require_strand=False) -> None:
        path, filename = os.path.split(vcf_file)
        output_path_prefix = '.'.join(filename.split('.')[:-1])
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = path

        output_path_prefix = os.path.join(output_dir, output_path_prefix)
        variants = read_vcf_file(
            vcf_file,
            strand_index=strand_index,
            require_strand=require_strand,
            output_NAs_to_file="{0}.NA".format(output_path_prefix),
            seq_context=(self._start_radius, self._end_radius),
            reference_sequence=self.reference_sequence)
        reporters = self._initialize_reporters(
            save_data,
            output_path_prefix,
            output_format,
            VARIANTEFFECT_COLS,
            output_size=len(variants),
            mode="varianteffect")

        batch_ref_seqs = []
        batch_alt_seqs = []
        batch_ids = []
        t_i = time()
        for ix, (chrom, pos, name, ref, alt, strand) in enumerate(variants):
            # centers the sequence containing the ref allele based on the size of ref
            center = pos + len(ref) // 2
            start = center - self._start_radius
            end = center + self._end_radius
            ref_sequence_encoding, contains_unk = \
                self.reference_sequence.get_encoding_from_coords_check_unk(
                    chrom, start, end)

            ref_encoding = self.reference_sequence.sequence_to_encoding(ref)
            alt_sequence_encoding = _process_alt(
                chrom, pos, ref, alt, start, end,
                ref_sequence_encoding,
                self.reference_sequence)

            match = True
            seq_at_ref = None
            if len(ref) and len(ref) < self.sequence_length:
                match, ref_sequence_encoding, seq_at_ref = _handle_standard_ref(
                    ref_encoding,
                    ref_sequence_encoding,
                    self.sequence_length,
                    self.reference_sequence)
            elif len(ref) >= self.sequence_length:
                match, ref_sequence_encoding, seq_at_ref = _handle_long_ref(
                    ref_encoding,
                    ref_sequence_encoding,
                    self._start_radius,
                    self._end_radius,
                    self.reference_sequence)

            if contains_unk:
                warnings.warn("For variant ({0}, {1}, {2}, {3}, {4}, {5}), "
                              "reference sequence contains unknown base(s)"
                              "--will be marked `True` in the `contains_unk` column "
                              "of the .tsv or the row_labels .txt file."
                              .format(chrom, pos, name, ref, alt, strand))
            if not match:
                warnings.warn("For variant ({0}, {1}, {2}, {3}, {4}, {5}), "
                              "reference does not match the reference genome. "
                              "Reference genome contains {6} instead. "
                              "Predictions/scores associated with this "
                              "variant--where we use '{3}' in the input "
                              "sequence--will be marked `False` in the `ref_match` "
                              "column of the .tsv or the row_labels .txt file"
                              .format(chrom, pos, name, ref, alt, strand, seq_at_ref))
            batch_ids.append((chrom, pos, name, ref, alt, strand, match, contains_unk))
            if strand == '-':
                ref_sequence_encoding = get_reverse_complement_encoding(
                    ref_sequence_encoding,
                    self.reference_sequence.BASES_ARR,
                    self.reference_sequence.COMPLEMENTARY_BASE_DICT)
                alt_sequence_encoding = get_reverse_complement_encoding(
                    alt_sequence_encoding,
                    self.reference_sequence.BASES_ARR,
                    self.reference_sequence.COMPLEMENTARY_BASE_DICT)
            batch_ref_seqs.append(ref_sequence_encoding)
            batch_alt_seqs.append(alt_sequence_encoding)

            if len(batch_ref_seqs) >= self.batch_size:
                _handle_ref_alt_predictions(
                    self.model,
                    batch_ref_seqs,
                    batch_alt_seqs,
                    batch_ids,
                    reporters,
                    use_cuda=self.use_cuda)
                batch_ref_seqs = []
                batch_alt_seqs = []
                batch_ids = []

            if ix and ix % 10000 == 0:
                print("[STEP {0}]: {1} s to process 10000 variants.".format(
                    ix, time() - t_i))
                t_i = time()

        if batch_ref_seqs:
            _handle_ref_alt_predictions(
                self.model,
                batch_ref_seqs,
                batch_alt_seqs,
                batch_ids,
                reporters,
                use_cuda=self.use_cuda)

        for r in reporters:
            r.write_to_file()

    def _pad_or_truncate_sequence(self, sequence: str) -> str:
        if len(sequence) < self.sequence_length:
            sequence = _pad_sequence(
                sequence,
                self.sequence_length,
                self.reference_sequence.UNK_BASE,
            )
        elif len(sequence) > self.sequence_length:
            sequence = _truncate_sequence(sequence, self.sequence_length)
        return sequence
