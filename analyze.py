from skylar.predict import AnalyzeSequences
from skylar.sequences import Genome
from skylar.utils import DeeperDeepSEA
from skylar.utils import load_features_list
from skylar.utils import NonStrandSpecific

model_architecture = NonStrandSpecific(DeeperDeepSEA(1000, 919))

# OSError: SavedModel file does not exist at: analyzing/example_deeperdeepsea.pth.tar/{saved_model.pbtxt|saved_model.pb}

features = load_features_list("analyzing/distinct_features.txt")
analysis = AnalyzeSequences(
    model_architecture,
    "analyzing/example_deeperdeepsea.pth.tar",
    sequence_length=1000,
    features=features,
    use_cuda=False,
    reference_sequence=Genome("analyzing/sequences.fasta"))
analysis.in_silico_mutagenesis_from_file("analyzing/sequences.fasta",
                                         save_data=["abs_diffs", "logits", "predictions"],
                                         output_dir="./analyzing",
                                         use_sequence_name=False)
