import os
import importlib
import sys
from time import strftime
from types import ModuleType
from typing import Dict

import tensorflow as tf

from . import _is_lua_trained_model
from . import instantiate


def class_instantiate(classobj):
    for attr, obj in classobj.__dict__.items():
        is_module = getattr(obj, '__module__', None)
        if is_module and "selene_sdk" in is_module and attr != "model":
            class_instantiate(obj)
    classobj.__init__(**classobj.__dict__)


def module_from_file(path: str) -> ModuleType:
    parent_path, module_file = os.path.split(path)
    loader = importlib.machinery.SourceFileLoader(module_file[:-3], path)
    module = ModuleType(loader.name)
    loader.exec_module(module)
    return module


def module_from_dir(path: str):
    parent_path, module_dir = os.path.split(path)
    sys.path.insert(0, parent_path)
    return importlib.import_module(module_dir)


def initialize_model(model_configs: Dict, train: bool = True, learning_rate: float = None):
    import_model_from = model_configs["path"]
    model_class_name = model_configs["class"]

    if os.path.isdir(import_model_from):
        module: ModuleType = module_from_dir(import_model_from)
    else:
        module: ModuleType = module_from_file(import_model_from)
    model_class = getattr(module, model_class_name)

    model = model_class(**model_configs["class_args"])
    if "non_strand_specific" in model_configs:
        from selene_sdk.utils import NonStrandSpecific
        model = NonStrandSpecific(model, mode=model_configs["non_strand_specific"])

    _is_lua_trained_model(model)
    criterion = module.criterion()
    if train:
        optim_class, optim_kwargs = module.get_optimizer(learning_rate)
        return model, criterion, optim_class, optim_kwargs
    else:
        return model, criterion


def execute(operations, configs: Dict, output_dir):
    model = None
    train_model = None
    for op in operations:
        if op == "train":
            model, loss, optim, optim_kwargs = initialize_model(
                configs["model"], learning_rate=configs["learning_rate"])

            sampler_info = configs["sampler"]
            if output_dir is not None:
                sampler_info.bind(output_dir=output_dir)
            sampler = instantiate(sampler_info)
            train_model_info = configs["train_model"]
            train_model_info.bind(model=model,
                                  data_sampler=sampler,
                                  loss_criterion=loss,
                                  optimizer_class=optim,
                                  optimizer_kwargs=optim_kwargs)
            if output_dir is not None:
                train_model_info.bind(output_dir=output_dir)

            train_model = instantiate(train_model_info)
            if "load_test_set" in configs and configs["load_test_set"] and "evaluate" in operations:
                train_model.create_test_set()
            train_model.train_and_validate()

        elif op == "evaluate":
            if train_model is not None:
                train_model.evaluate()

            if not model:
                model, loss = initialize_model(configs["model"], train=False)
            if "evaluate_model" in configs:
                sampler_info = configs["sampler"]
                sampler = instantiate(sampler_info)
                evaluate_model_info = configs["evaluate_model"]
                evaluate_model_info.bind(model=model, criterion=loss, data_sampler=sampler)
                if output_dir is not None:
                    evaluate_model_info.bind(output_dir=output_dir)

                evaluate_model = instantiate(evaluate_model_info)
                evaluate_model.evaluate()

        elif op == "analyze":
            if not model:
                model, _ = initialize_model(configs["model"], train=False)
            analyze_seqs_info = configs["analyze_sequences"]
            analyze_seqs_info.bind(model=model)

            analyze_seqs = instantiate(analyze_seqs_info)
            if "variant_effect_prediction" in configs:
                vareff_info = configs["variant_effect_prediction"]
                if "vcf_files" not in vareff_info:
                    raise ValueError("variant effect prediction requires as input a list of "
                                     "1 or more *.vcf files ('vcf_files').")
                for filepath in vareff_info.pop("vcf_files"):
                    analyze_seqs.variant_effect_prediction(filepath, **vareff_info)
            if "in_silico_mutagenesis" in configs:
                ism_info = configs["in_silico_mutagenesis"]
                if "sequence" in ism_info:
                    analyze_seqs.in_silico_mutagenesis(**ism_info)
                elif "input_path" in ism_info:
                    analyze_seqs.in_silico_mutagenesis_from_file(**ism_info)
                elif "fa_files" in ism_info:
                    for filepath in ism_info.pop("fa_files"):
                        analyze_seqs.in_silico_mutagenesis_from_file(filepath, **ism_info)
                else:
                    raise ValueError("in silico mutagenesis requires as input "
                                     "the path to the FASTA file "
                                     "('input_path') or a sequence "
                                     "('input_sequence') or a list of "
                                     "FASTA files ('fa_files'), but found "
                                     "neither.")
            if "prediction" in configs:
                predict_info = configs["prediction"]
                analyze_seqs.get_predictions(**predict_info)


def parse_configs_and_run(configs: Dict, create_subdirectory: bool = True, learning_rate: float = None):
    operations = configs["ops"]

    if "train" in operations and "learning_rate" not in configs and learning_rate != "None":
        configs["learning_rate"] = float(learning_rate)
    elif "train" in operations and "learning_rate" in configs and learning_rate != "None":
        print("Warning: learning rate specified in both the "
              "configuration dict and this method's `learning_rate` parameter. "
              "Using the `learning_rate` value input to `parse_configs_and_run` "
              "({0}, not {1}).".format(learning_rate, configs["learning_rate"]))

    current_run_output_dir = None
    if "output_dir" not in configs and ("train" in operations or "evaluate" in operations):
        print("No top-level output directory specified. All constructors "
              "to be initialized (e.g. Sampler, TrainModel) that require "
              "this parameter must have it specified in their individual "
              "parameter configuration.")
    elif "output_dir" in configs:
        current_run_output_dir = configs["output_dir"]
        os.makedirs(current_run_output_dir, exist_ok=True)
        if "create_subdirectory" in configs:
            create_subdirectory = configs["create_subdirectory"]
        if create_subdirectory:
            current_run_output_dir = os.path.join(
                current_run_output_dir, strftime("%Y-%m-%d-%H-%M-%S"))
            os.makedirs(current_run_output_dir)
        print("Outputs and logs saved to {0}".format(current_run_output_dir))

    if "random_seed" in configs:
        tf.random.set_seed(configs["random_seed"])
    else:
        print("Warning: no random seed specified in config file. "
              "Using a random seed ensures results are reproducible.")
    execute(operations, configs, current_run_output_dir)
