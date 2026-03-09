import argparse
import os
from os import path

import torch

from bspline_curve_approximation_transformer.training import training_setup
from bspline_curve_approximation_transformer.utils import (get_default_parameters, parser_add_eval_args,
                                                           prep_param_eval_args, task_specific_defaults)


def prep_params(args) -> dict:
    default_object = get_default_parameters()

    default_object['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.snap:
        snapshot_object = torch.load(args.snap,
                                     map_location=None if torch.cuda.is_available() else default_object['device'])
        default_object.update(snapshot_object)
        default_object['snap'] = args.snap

    default_object = prep_param_eval_args(default_object, args)

    main_folder = path.split(path.split(args.snap)[0])[0]

    default_object['output_folder'] = path.join(main_folder, f"Inference_{path.split(args.snap)[1]}", path.split(default_object['dataset_root_dir'])[1])
    if default_object['custom_output_folder'] is not None:
        default_object['output_folder'] = path.join(main_folder, default_object['custom_output_folder'], path.split(default_object['dataset_root_dir'])[1])

    default_object['eval_only'] = True

    print(default_object['output_folder'])

    os.makedirs(default_object['output_folder'], exist_ok=True)

    default_object = task_specific_defaults(default_object)

    return default_object


def main():
    parser = argparse.ArgumentParser()

    parser = parser_add_eval_args(parser)

    args = parser.parse_args()

    training_setup(**prep_params(args))


if __name__ == "__main__":
    main()
