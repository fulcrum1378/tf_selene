from skylar.utils import load_path, parse_configs_and_run

parse_configs_and_run(load_path("training/config.yml"), learning_rate=0.01)
