import matplotlib.pyplot as plt

import skylar.interpret
from skylar.interpret import ISMResult
from skylar.sequences import Genome

ism = ISMResult.from_file("analyzing/0_predictions.tsv")
score_matrix = ism.get_score_matrix_for("K562|H3K27ac|None")[:50, ]

reference_encoding = Genome.sequence_to_encoding(ism.reference_sequence)[:50, ] == 1.
figure, (ax) = plt.subplots(1, 1, figsize=(10, 4))
ax.patch.set(edgecolor="lightgrey", hatch="//")
skylar.interpret.heatmap(score_matrix, mask=reference_encoding, cbar=True, ax=ax, linewidth=0.5)

figure, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 6))
skylar.interpret.sequence_logo(score_matrix, order="alpha", ax=ax1)
skylar.interpret.sequence_logo(score_matrix, order="value", ax=ax2)

rescaled_score_matrix = skylar.interpret.rescale_score_matrix(score_matrix, base_scaling="max_effect")
figure, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 6))
skylar.interpret.sequence_logo(rescaled_score_matrix, ax=ax1)
skylar.interpret.sequence_logo(score_matrix, ax=ax2)

# TypeError: only integer scalar arrays can be converted to a scalar index
# ATTENTION: This error is not about your "skylar" code
