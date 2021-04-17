from selene_sdk.utils import load_path
from selene_sdk.utils import parse_configs_and_run

configs = load_path("dataset/train.yaml")
parse_configs_and_run(configs, lr=0.01)
