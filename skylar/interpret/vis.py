from collections import defaultdict
from copy import deepcopy
import os
import re
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.patheffects
from matplotlib.text import TextPath
import pkg_resources
from plotly.offline import download_plotlyjs, plot
import plotly.graph_objs as go
import seaborn as sns
import tabix

from ..sequences import Genome

_SVG_PATHS = {'T': "M 0,100 l 100, 0 l 0,-25 l -37.5, 0 l 0,-75 l -25, 0 " +
                   "l 0,75 l -37.5,0 l 0,25 z",
              'C': ("M 100,12.5 l 0,25 c 0,0 -25,-15 -50,-12.5 " +
                    "c 0,0 -25,0 -25,25 c 0,0 0,25 25,25 c 0,0 25,2.5 50,-15" +
                    " l 0, 25 C 100,87.5 75,100 50,100 C 50,100 0,100 0,50 " +
                    "C 0,50 0,0 50,0 C 50,0 75,0 100,12.5 z"),
              'G': ("M 100,12.5 l 0,25 c 0,0 -25,-15 -50,-12.5 " +
                    "c 0,0 -25,0 -25,25 c 0,0 0,25 25,25 c 0,0 25,2.5 50,-15" +
                    " l 0, 25 C 100,87.5 75,100 50,100 C 50,100 0,100 0,50 " +
                    "C 0,50 0,0 50,0 C 50,0 75,0 100,12.5 M 100,37.5 " +
                    "l 0,17.5 l -50,0 l 0,-17 l 25,0 l 0,-25 l 25,0 z"),
              'A': ("M 0,0 l 37.5,100 l 25,0 l 37.5,-100 l -25,0 l -9.375,25" +
                    " l -31.25,0 l -9.375,-25 l -25,0 z 0,0 M 43.75, 50 " +
                    "l 12.5,0 l -5.859375,15.625 l -5.859375,-15.625 z"),
              'U': ("M 0,100 l 25,0 l 0,-50 C 25,50 25,25 50,25" +
                    " C 50,25 75,25 75,50 l 0,50 l 25,0 L 100,50 " +
                    "C 100,50 100,0, 50,0 C 50,0 0,0 0,50 l 0,50 z")}


def _svg_parse(path_string):
    commands = {'M': (Path.MOVETO,),
                'L': (Path.LINETO,),
                'Q': (Path.CURVE3,) * 2,
                'C': (Path.CURVE4,) * 3,
                'Z': (Path.CLOSEPOLY,)}
    path_re = re.compile(r'([MLHVCSQTAZ])([^MLHVCSQTAZ]+)', re.IGNORECASE)
    float_re = re.compile(r'(?:[\s,]*)([+-]?\d+(?:\.\d+)?)')
    vertices = []
    codes = []
    last = (0, 0)
    for cmd, values in path_re.findall(path_string):
        points = [float(v) for v in float_re.findall(values)]
        points = np.array(points).reshape((len(points) // 2, 2))
        if cmd.islower():
            points += last
        cmd = cmd.capitalize()
        if len(points) > 0:
            last = points[-1]
        codes.extend(commands[cmd])
        vertices.extend(points.tolist())
    return np.array(vertices), codes


for k in _SVG_PATHS.keys():
    _SVG_PATHS[k] = _svg_parse(_SVG_PATHS[k])


class _TextPathRenderingEffect(matplotlib.patheffects.AbstractPathEffect):
    def __init__(self, bar, x_translation=0., y_translation=0.,
                 x_scale=1., y_scale=1.):
        self._bar = bar
        self._x_translation = x_translation
        self._y_translation = y_translation
        self._x_scale = x_scale
        self._y_scale = y_scale

    def draw_path(self, renderer, gc, tpath, affine, rgbFace=None):
        b_x, b_y, b_w, b_h = self._bar.get_extents().bounds
        t_x, t_y, t_w, t_h = tpath.get_extents().bounds
        translation = [b_x - t_x, b_y - t_y]
        translation[0] += self._x_translation
        translation[1] += self._y_translation
        scale = [b_w / t_w, b_h / t_h]
        scale[0] *= self._x_scale
        scale[1] *= self._y_scale
        affine = affine.identity().scale(*scale).translate(*translation)
        renderer.draw_path(gc, tpath, affine, rgbFace)


def sequence_logo(score_matrix, order="value", width=1.0, ax=None,
                  sequence_type=Genome, font_properties=None,
                  color_scheme=None,
                  **kwargs):
    # Note that everything will break if we do not deepcopy.
    score_matrix = deepcopy(score_matrix)

    score_matrix = score_matrix.transpose()
    if font_properties is not None:
        warnings.warn(
            "Specifying a value for `font_properties` (other than `None`) "
            "will use the `matplotlib`-based character paths, and causes "
            "distortions in the plotted motif. We recommend leaving "
            "`font_properties=None`. See the documentation for details.",
            UserWarning)

    if color_scheme is None:
        color_scheme = sns.color_palette("Set1",
                                         n_colors=len(sequence_type.BASES_ARR))
        color_scheme = color_scheme.as_hex()
    if len(color_scheme) < len(sequence_type.BASES_ARR):
        raise ValueError(
            "Color scheme is shorter than number of bases in sequence.")

    if score_matrix.shape[0] != len(sequence_type.BASES_ARR):
        raise ValueError(
            "Got score with {0} bases for sequence with {1} bases.".format(
                score_matrix.shape[0], len(sequence_type.BASES_ARR)))
    if ax is None:
        _, ax = plt.subplots(figsize=score_matrix.shape)

    # Determine offsets depending on sort order.
    positive_offsets = np.zeros_like(score_matrix)
    negative_offsets = np.zeros_like(score_matrix)
    bases = np.empty(score_matrix.shape, dtype=object)
    bases[:, :] = "?"  # This ensures blanks are visually obvious.

    # Change ordering of things based on input arguments.
    if order == "alpha":
        for i in range(score_matrix.shape[0]):
            bases[i, :] = sequence_type.BASES_ARR[i]

    elif order == "value":
        if np.sum(score_matrix < 0) != 0:
            sorted_scores = np.zeros_like(score_matrix)
            for j in range(score_matrix.shape[1]):
                # Sort the negative values and put them at bottom.
                div = np.sum(score_matrix[:, j] < 0.)
                negative_idx = np.argwhere(score_matrix[:, j] < 0.).flatten()
                negative_sort_idx = np.argsort(score_matrix[negative_idx, j],
                                               axis=None)
                sorted_scores[:div, j] = score_matrix[
                    negative_idx[negative_sort_idx], j]
                bases[:div, j] = sequence_type.BASES_ARR[
                    negative_idx[negative_sort_idx]].flatten()

                # Sort the positive values and stack atop the negatives.
                positive_idx = np.argwhere(score_matrix[:, j] >= 0.).flatten()
                positive_sort_idx = np.argsort(score_matrix[positive_idx, j],
                                               axis=None)
                sorted_scores[div:, j] = score_matrix[
                    positive_idx[positive_sort_idx], j]
                bases[div:, j] = sequence_type.BASES_ARR[
                    positive_idx[positive_sort_idx]].flatten()
            score_matrix = sorted_scores
        else:
            for j in range(score_matrix.shape[1]):
                sort_idx = np.argsort(score_matrix[:, j], axis=None)[::-1]
                bases[:, j] = sequence_type.BASES_ARR[sort_idx]
                score_matrix[:, j] = score_matrix[sort_idx, j]

    # Create offsets for each bar.
    for i in range(score_matrix.shape[0] - 1):
        y_coords = score_matrix[i, :]
        if i > 0:
            negative_offsets[i + 1, :] = negative_offsets[i, :]
            positive_offsets[i + 1, :] = positive_offsets[i, :]
        neg_idx = np.argwhere(y_coords < 0.)
        pos_idx = np.argwhere(y_coords >= 0.)
        negative_offsets[i + 1, neg_idx] += y_coords[neg_idx]
        positive_offsets[i + 1, pos_idx] += y_coords[pos_idx]

    for i in range(score_matrix.shape[0]):
        x_coords = np.arange(score_matrix.shape[1]) + 0.5
        y_coords = score_matrix[i, :]

        # Manage negatives and positives separately.
        offsets = np.zeros(score_matrix.shape[1])
        negative_idx = np.argwhere(y_coords < 0.)
        positive_idx = np.argwhere(y_coords >= 0.)
        offsets[negative_idx] = negative_offsets[i, negative_idx]
        offsets[positive_idx] = positive_offsets[i, positive_idx]
        bars = ax.bar(x_coords, y_coords, color="black", width=width,
                      bottom=offsets)
        for j, bar in enumerate(bars):
            base = bases[i, j]
            bar.set_color(color_scheme[sequence_type.BASE_TO_INDEX[base]])
            bar.set_edgecolor(None)

    # Iterate over the barplot's bars and turn them into letters.
    new_patches = []
    for i, bar in enumerate(ax.patches):
        base_idx = i // score_matrix.shape[1]
        seq_idx = i % score_matrix.shape[1]
        base = bases[base_idx, seq_idx]
        # We construct a text path that tracks the bars in the barplot.
        # Thus, the barplot takes care of scaling and translation,
        #  and we just copy it.
        if font_properties is None:
            text = Path(_SVG_PATHS[base][0], _SVG_PATHS[base][1])
        else:
            text = TextPath((0., 0.), base, fontproperties=font_properties)
        b_x, b_y, b_w, b_h = bar.get_extents().bounds
        t_x, t_y, t_w, t_h = text.get_extents().bounds
        scale = (b_w / t_w, b_h / t_h)
        translation = (b_x - t_x, b_y - t_y)
        text = PathPatch(text, facecolor=bar.get_facecolor(), lw=0.)
        bar.set_facecolor("none")
        text.set_path_effects([_TextPathRenderingEffect(bar)])
        transform = transforms.Affine2D().translate(*translation).scale(*scale)
        text.set_transform(transform)
        new_patches.append(text)

    for patch in new_patches:
        ax.add_patch(patch)
    ax.set_xlim(0, score_matrix.shape[1])
    ax.set_xticks(np.arange(score_matrix.shape[1]) + 0.5)
    ax.set_xticklabels(np.arange(score_matrix.shape[1]))
    return ax


def rescale_score_matrix(score_matrix, base_scaling="identity",
                         position_scaling="identity"):
    # Note that things can break if we do not deepcopy.
    score_matrix = deepcopy(score_matrix)

    score_matrix = score_matrix.transpose()
    rescaled_scores = score_matrix

    # Scale individual bases.
    if base_scaling == "identity" or base_scaling == "probability":
        pass
    elif base_scaling == "max_effect":
        rescaled_scores = score_matrix - np.min(score_matrix, axis=0)
    else:
        raise ValueError(
            "Could not find base scaling \"{0}\".".format(base_scaling))

    # Scale each position
    if position_scaling == "max_effect":
        max_effects = np.max(score_matrix, axis=0) - np.min(score_matrix,
                                                            axis=0)
        rescaled_scores /= rescaled_scores.sum(axis=0)[np.newaxis, :]
        rescaled_scores *= max_effects[np.newaxis, :]
    elif position_scaling == "probability":
        rescaled_scores /= np.sum(score_matrix, axis=0)[np.newaxis, :]
    elif position_scaling != "identity":
        raise ValueError(
            "Could not find position scaling \"{0}\".".format(
                position_scaling))
    return rescaled_scores.transpose()


def heatmap(score_matrix, mask=None, sequence_type=Genome, **kwargs):
    # Note that some things can break if we do not deepcopy.
    score_matrix = deepcopy(score_matrix)

    # This flipping is so that ordering is consistent with ordering
    # in the sequence logo.
    if mask is not None:
        mask = mask.transpose()
        mask = np.flip(mask, axis=0)
    score_matrix = score_matrix.transpose()
    score_matrix = np.flip(score_matrix, axis=0)

    if "yticklabels" in kwargs:
        yticklabels = kwargs.pop("yticklabels")
    else:
        yticklabels = sequence_type.BASES_ARR[::-1]
    if "cbar_kws" in kwargs:
        cbar_kws = kwargs.pop("cbar_kws")
    else:
        cbar_kws = dict(use_gridspec=False, location="bottom", pad=0.2)
    if "cmap" in kwargs:
        cmap = kwargs.pop("cmap")
    else:
        cmap = "Blues"
    ret = sns.heatmap(score_matrix, mask=mask, yticklabels=yticklabels,
                      cbar_kws=cbar_kws, cmap=cmap, **kwargs)
    ret.set_yticklabels(labels=ret.get_yticklabels(), rotation=0)
    return ret


def load_variant_abs_diff_scores(input_path):
    features = []
    labels = []
    diffs = []
    with open(input_path, 'r') as file_handle:
        colnames = file_handle.readline()
        features = colnames.strip().split('\t')[5:]
        for line in file_handle:
            cols = line.strip().split('\t')
            scores = [float(f) for f in cols[5:]]
            label = tuple(cols[:5])
            diffs.append(scores)
            labels.append(label)
    diffs = np.array(diffs)
    return diffs, labels, features


def sort_standard_chrs(chrom):
    chrom = chrom[3:]
    if chrom.isdigit():
        return int(chrom)
    if chrom == 'X':
        return 23
    elif chrom == 'Y':
        return 24
    elif chrom == 'M':
        return 25
    else:  # unknown chr
        return 26


def ordered_variants_and_indices(labels):
    labels_dict = defaultdict(list)
    for i, l in enumerate(labels):
        chrom, pos, name, ref, alt = l
        pos = int(pos)
        info = (i, pos, ref, alt)
        labels_dict[chrom].append(info)
    for chrom, labels_list in labels_dict.items():
        labels_list.sort(key=lambda tup: tup[1:])
    ordered_keys = sorted(labels_dict.keys(),
                          key=sort_standard_chrs)

    ordered_labels = []
    ordered_label_indices = []
    for chrom in ordered_keys:
        for l in labels_dict[chrom]:
            index, pos, ref, alt = l
            ordered_label_indices.append(index)
            ordered_labels.append((chrom, pos, ref, alt))
    return ordered_labels, ordered_label_indices


def _label_tuple_to_text(label, diff, genes=None):
    chrom, pos, ref, alt = label
    if genes is not None:
        if len(genes) == 0:
            genes_str = "none found"
        else:
            genes_str = ', '.join(genes)
        text = ("max diff score: {0}<br />{1} {2}, {3}/{4}<br />"
                "closest protein-coding gene(s): {5}").format(
            diff, chrom, pos, ref, alt, genes_str)
    else:
        text = "max diff score: {0}<br />{1} {2}, {3}/{4}".format(
            diff, chrom, pos, ref, alt)
    return text


def _variant_closest_genes(label, tabix_fh, chrs_gene_intervals):
    chrom = label[0]
    pos = label[1]
    closest_genes = []
    try:
        overlaps = tabix_fh.query(chrom, pos, pos + 1)
        for o in overlaps:
            closest_genes.append(o[-1])
    except tabix.TabixError:
        pass
    if len(closest_genes) != 0:
        return closest_genes
    gene_intervals = chrs_gene_intervals[chrom]
    closest = None
    for (start, end, strand, gene) in gene_intervals:
        if start < pos and closest and closest == pos - end:
            closest_genes.append(gene)
        elif start < pos and (closest is None
                              or closest > pos - end):
            closest = pos - end
            closest_genes = [gene]
        elif start > pos and closest and closest == start - pos:
            closest_genes.append(gene)
        elif start > pos and (closest is None
                              or closest > start - pos):
            closest = start - pos
            closest_genes = [gene]
    return closest_genes


def _load_chrs_gene_intervals(gene_intervals_bed):
    chrs_gene_intervals = defaultdict(list)
    with open(gene_intervals_bed, 'r') as file_handle:
        for line in file_handle:
            cols = line.strip().split('\t')
            chrom = cols[0]
            start = int(cols[1])
            end = int(cols[2])
            strand = cols[3]
            gene = cols[4]
            chrs_gene_intervals[chrom].append((start, end, strand, gene))
    return chrs_gene_intervals


def _variants_closest_protein_coding_gene(labels, version="hg38"):
    gene_intervals_tabix = pkg_resources.resource_filename(
        "skylar",
        ("interpret/data/gencode_v28_{0}/"
         "protein_coding_l12_genes.bed.gz").format(version))
    gene_intervals_bed = pkg_resources.resource_filename(
        "skylar",
        ("interpret/data/gencode_v28_{0}/"
         "protein_coding_l12_genes.bed").format(version))

    tabix_fh = tabix.open(gene_intervals_tabix)
    chrs_gene_intervals = _load_chrs_gene_intervals(gene_intervals_bed)

    labels_gene_information = []
    for l in labels:
        labels_gene_information.append(_variant_closest_genes(
            l, tabix_fh, chrs_gene_intervals))

    return labels_gene_information


def variant_diffs_scatter_plot(data,
                               labels,
                               features,
                               output_path,
                               filter_features=None,
                               labels_sort_fn=ordered_variants_and_indices,
                               nth_percentile=None,
                               hg_reference_version=None,
                               threshold_line=None,
                               auto_open=False):
    labels_ordered, label_indices = labels_sort_fn(labels)
    variant_closest_genes = None
    if hg_reference_version is not None:
        variant_closest_genes = _variants_closest_protein_coding_gene(
            labels_ordered, version=hg_reference_version)
    ordered_data = data[label_indices, :]

    feature_indices = None
    if filter_features is not None:
        feature_indices = filter_features(features)
        ordered_data = data[:, feature_indices]
    variants_max_diff = np.amax(ordered_data, axis=1)

    display_labels = None
    if nth_percentile:
        p = np.percentile(variants_max_diff, nth_percentile)
        keep = np.where(variants_max_diff >= p)[0]
        print("{0} variants with max abs diff score above {1} are in the "
              "{2}th percentile.".format(len(keep), p, nth_percentile))
        variants_max_diff = variants_max_diff[keep]
        display_labels = []
        for i, l in enumerate(labels_ordered):
            if i not in keep:
                continue
            display_labels.append(l)
    else:
        display_labels = labels_ordered

    if variant_closest_genes:
        text_labels = [
            _label_tuple_to_text(l, d, g) for l, d, g in
            zip(display_labels, variants_max_diff, variant_closest_genes)]
    else:
        text_labels = [_label_tuple_to_text(l, d) for l, d in
                       zip(display_labels, variants_max_diff)]

    label_x = [' '.join([l[0], str(l[1])]) for l in display_labels]
    data = [go.Scatter(x=label_x,
                       y=variants_max_diff,
                       mode='markers',
                       marker=dict(
                           color="#39CCCC",
                           line=dict(width=1)),
                       text=text_labels,
                       hoverinfo="text")]

    go_layout = {
        "title": "Max probability difference scores",
        "hovermode": "closest",
        "hoverlabel": {
            "font": {"size": 16}
        },
        "xaxis": {
            "title": "Genome coordinates",
            "showticklabels": True,
            "tickangle": 35,
            "titlefont": {"family": "Arial, sans-serif",
                          "size": 16},
            "nticks": 25,
            "tickmode": "auto",
            "automargin": True
        },
        "yaxis": {
            "title": "Absolute difference",
            "titlefont": {"family": "Arial, sans-serif",
                          "size": 16}
        }
    }
    if isinstance(threshold_line, float):
        layout = go.Layout(
            **go_layout,
            shapes=[
                {"type": "line",
                 "x0": label_x[0],
                 "y0": threshold_line,
                 "x1": label_x[-1],
                 "y1": threshold_line,
                 "line": {
                     "color": "rgba(255, 99, 71, 1)",
                     "width": 2
                 }
                 }
            ])
    else:
        layout = go.Layout(**go_layout)
    fig = go.Figure(data=data, layout=layout)
    path, filename = os.path.split(output_path)
    os.makedirs(path, exist_ok=True)
    plot(fig, filename=output_path, auto_open=auto_open)
    return fig
