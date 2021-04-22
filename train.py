from selene_sdk.utils import load_path, parse_configs_and_run

parse_configs_and_run(load_path("training/config.yml"), lr=0.01)
