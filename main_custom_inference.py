import numpy as np

from bspline_curve_approximation_transformer.api import call_custom_inference


def main_custom_inference():
    data_bspline = np.random.random((10,100,2))
    call_custom_inference(data_bspline=data_bspline)


if __name__ == "__main__":
    main_custom_inference()
