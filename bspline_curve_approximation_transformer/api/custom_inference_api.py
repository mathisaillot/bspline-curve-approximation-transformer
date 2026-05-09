import argparse
import os

import numpy as np
import torch

from bspline_curve_approximation_transformer.training import custom_inference
from bspline_curve_approximation_transformer.utils import get_default_parameters, prep_param_custom_inference_args, \
    task_specific_defaults, CustomInferenceArgs, parser_add_custom_inference_args


def prep_params_custom_inference(data_bspline: np.ndarray, args) -> dict:
    default_object = get_default_parameters()

    default_object['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.snap:
        snapshot_object = torch.load(args.snap,
                                     map_location=None if torch.cuda.is_available() else default_object['device'])
        default_object.update(snapshot_object)
        default_object['snap'] = args.snap

    default_object = prep_param_custom_inference_args(default_object, args)

    default_object['eval_only'] = True

    if default_object['output_folder'] is None:
        default_object['output_folder'] = "output"

    print(default_object['output_folder'])

    os.makedirs(default_object['output_folder'], exist_ok=True)

    default_object = task_specific_defaults(default_object)

    default_object['data_bspline'] = data_bspline

    return default_object

def call_custom_inference_args(data_bspline: np.ndarray, args: dict):
    return custom_inference(**prep_params_custom_inference(data_bspline, CustomInferenceArgs(args)))

def call_custom_inference(data_bspline: np.ndarray):
    parser = argparse.ArgumentParser()
    parser = parser_add_custom_inference_args(parser)
    args = parser.parse_args()
    return custom_inference(**prep_params_custom_inference(data_bspline, args))