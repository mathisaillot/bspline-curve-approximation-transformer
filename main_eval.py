import argparse
from os import path

import torch

from bspline_curve_approximation_transformer.training import training_setup
from bspline_curve_approximation_transformer.utils import (parser_add_network_args, parser_add_data_args,
                                                           parser_add_utils_args, parser_add_training_args,
                                                           get_default_parameters, prep_param_utils_args,
                                                           prep_param_training_args, prep_param_data_args,
                                                           prep_param_network_args, get_training_name,
                                                           parser_add_eval_args, prep_param_eval_args)


def prep_params(args) -> dict:
    default_object = get_default_parameters()

    default_object['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.snap:
        snapshot_object = torch.load(args.snap,
                                     map_location=None if torch.cuda.is_available() else default_object['device'])
        default_object.update(snapshot_object)
        default_object['snap'] = args.snap

    default_object = prep_param_eval_args(default_object, args)

    default_object['output_folder'] = path.split(path.split(args.snap)[0])[0]
    default_object['eval_only'] = True
    default_object['epochs'] = 1
    default_object['starting_epoch'] = 0
    default_object['step'] = 1

    return default_object


def main():
    parser = argparse.ArgumentParser()

    parser = parser_add_eval_args(parser)

    args = parser.parse_args()

    training_setup(**prep_params(args))


if __name__ == "__main__":
    main()
