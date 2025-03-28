import argparse

import torch

from bspline_curve_approximation_transformer.training import training_setup
from bspline_curve_approximation_transformer.utils import (parser_add_network_args, parser_add_data_args,
                                                           parser_add_utils_args, parser_add_training_args,
                                                           get_default_parameters, prep_param_utils_args,
                                                           prep_param_training_args, prep_param_data_args,
                                                           prep_param_network_args, get_training_name)


def prep_params(args) -> dict:
    default_object = get_default_parameters()

    default_object['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    default_object['output_folder'] = args.output_folder

    if args.snap:
        snapshot_object = torch.load(args.snap,
                                     map_location=None if torch.cuda.is_available() else default_object['device'])
        default_object.update(snapshot_object)
        if args.no_cache_model:
            default_object['model_parameters'] = None
        if args.no_cache_scheduler:
            default_object['optimizer_cache'] = None
            default_object['scheduler_cache'] = None
            default_object['starting_epoch'] = 0
        default_object['snap'] = args.snap

    default_object = prep_param_utils_args(default_object, args)
    default_object = prep_param_training_args(default_object, args)
    default_object = prep_param_data_args(default_object, args)
    default_object = prep_param_network_args(default_object, args)

    default_object['training_name'] = get_training_name(default_object, args)

    return default_object


def main():
    parser = argparse.ArgumentParser()

    parser = parser_add_training_args(parser)
    parser = parser_add_data_args(parser)
    parser = parser_add_network_args(parser)
    parser = parser_add_utils_args(parser)

    args = parser.parse_args()

    training_setup(**prep_params(args))


if __name__ == "__main__":
    main()
