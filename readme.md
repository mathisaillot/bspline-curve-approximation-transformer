# B-spline Curve Approximation With Transformer Neural Networks

Dataset and code snippets of ["B-spline Curve Approximation With Transformer Neural Networks"](https://hal.science/hal-04931216v1) article.


## B-spline Dataset

The dataset used for training can be found in the [bspline_dataset](bspline_dataset/readme.md) folder.
It contains `160,000` cubic bspline curves randomly generated.

## Code

Source code is located in [bspline-curve-approximation-transformer]() folder.

### Setup

To use this code base you need to setup a Python 3 virtual environment.
Then install [pytorch](https://pytorch.org), with or without CuDA.

You will then need to install the following python packages :
```bash
pip install numpy pandas torchmetrics transformers einops
```

### Training

Training a model can be done using [main_training.py](). 
This program takes a lot of parameters that can be used to change the different training and network parameters.

By default, the results, logs and checkpoints will be stored in [output]() folder.

### Evaluation

Once a model has been trained you can evaluate it using [main_eval.py](). 
This should not be necessary when trained using [main_training.py]().

## Model Checkpoints

Pretrained checkpoints will be available soon as external download.

## Citation

```bibtex
@article{saillot:hal-04931216,
  TITLE = {{B-spline curve approximation with transformer neural networks}},
  AUTHOR = {Saillot, Mathis and Michel, Dominique and Zidna, Ahmed},
  URL = {https://hal.science/hal-04931216},
  JOURNAL = {{Mathematics and Computers in Simulation}},
  PUBLISHER = {{Elsevier}},
  VOLUME = {223},
  PAGES = {275-287},
  YEAR = {2024},
  MONTH = Sep,
  DOI = {10.1016/j.matcom.2024.04.010},
  KEYWORDS = {B-spline ; Curve approximation ; Transformer neural network ; Knot vector prediction},
  PDF = {https://hal.science/hal-04931216v1/file/1-s2.0-S0378475424001368-main.pdf},
  HAL_ID = {hal-04931216},
  HAL_VERSION = {v1},
}
```