RDG README File Template --- General --- Version: 0.1 (2022-10-04)

This README file was generated on [2025-03-28] by [Mathis Saillot].
Last updated: [2025-03-28].

# GENERAL INFORMATION

## Dataset title: Pretrained Transformers of *B-spline Curve Approximation With Transformer Neural Networks* article

## DOI: TBD

### Contact email: mathis.saillot@univ-lorraine.fr

# DATA & FILE OVERVIEW

Model checkpoints along with configuration files and logs are grouped in folders for each training sessions.
Folder names are normalized to give a first idea of the parameters used to train the network.
Identical training can be distinguished using the date.

| **Name section**                   | **Description**                                                                                                                                               | **Required** | **Since Version** |
|------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|-------------------|
| `BSplineV{X.Y}`                    | Start of the name with `{X.Y}` the version of the codebase when trained                                                                                       | YES          | `V1.0`            |
| `{SURNAME}`                        | Custom `{SURNAME}` string used to personalised the folder name                                                                                                | NO           | `V1.0`            |
| `_{TL}TL`                          | `{TL}` is the `target_length` parameter; i.e. indicates the length of the data sequences used during training.                                                | YES          | `V1.0`            |
| `_RC`                              | Appears if the `relative_coord` parameter is used for the network; i.e. use of relative coordinates for input sequence.                                       | NO           | `V1.0`            |
| `_S`                               | Appears if the `shifting` parameter is used during training; i.e. use the shifting data augmentation during training.                                         | NO           | `V1.0`            |
| `_ST`                              | Appears if the `shared_token` parameter is used for the network; i.e. points coordinates and parameters are stacked in the depth dimension of input.          | NO           | `V1.0`            |
| `_T`                               | Appears if the `add_time` parameter is used for the network; i.e. concatenate time parameters to the input coordinates.                                       | NO           | `V1.0`            |
| `_N`                               | Appears if the `norm` parameter is used for the network; i.e. normalise input data by standardizing coordinates.                                              | NO           | `V1.0`            |
| `_U{stack}`                        | `{stack}` indicates the number of output heads when the `stack_u` parameter is used for the network.                                                          | NO           | `V1.0`            |
| `_SL`                              | Appears if the `scale_loss` parameter is used during training; i.e. applies a scaling to the different output heads loss depending on the size of the output. | NO           | `V1.0`            |
| `_SMX`                             | Appears if the `softmax` parameter is used; i.e. the softmax function is applied to the output head.                                                          | NO           | `V1.0`            |
| `_ADM`                             | Appears if the `adam` parameter is used during training; i.e. the AdamW optimizer during training instead of standard SGD.                                    | NO           | `V1.0`            |
| `_REG`                             | Appears if the `model_type` parameter is set to `regression`.                                                                                                 | NO           | `V1.0`            |
| `_CLS`                             | Appears if the `model_type` parameter is set to `clstoken`.                                                                                                   | NO           | `V1.0`            |
| `_LPE`                             | Appears if the `learnable_pe` parameter is used for the network; i.e. use learnable positional embedding layer at the start of the network.                   | NO           | `V1.0`            |
| `_NPE`                             | Appears if the `no_pe` parameter is used for the network; i.e. do not use any positional embedding layer at the start of the network.                         | NO           | `V1.0`            |
| `_{H}x{D}`                         | `{H}` is the number of heads for each of the `{D}` self-attention layers of the network, set by the `n_heads` and `depth` parameters.                         | YES          | `V1.0`            |
| `_{DH}DH`                          | `{DH}` is the embedding dimension of each self-attention head set by the `dim_head` parameter.                                                                | YES          | `V1.0`            |
| `_{HSZ}HSZ`                        | `{HSZ}` is the embedding dimension of the network set by the `hidden_size` parameter.                                                                         | YES          | `V1.0`            |
| `_{BS}`                            | `{BS}` is the `batch_size` parameter used during training.                                                                                                    | YES          | `V1.0`            |
| `_{EPX}`                           | `{EPX}` is the number of `epochs` of the training.                                                                                                            | YES          | `V1.0`            |
| `_{STEP}`                          | `{STEP}` is the number of `step` of the learning rate scheduler used during training.                                                                         | YES          | `V1.0`            |
| `_{LR}`                            | `{LR}` is the base learning rate used during training set by the `lr` parameter.                                                                              | YES          | `V1.0`            |
| `_{GMA}`                           | `{GMA}` is the `gamma` parameter used for the step learning rate scheduler during training.                                                                   | YES          | `V1.0`            |
| `_{MMT}`                           | `{MMT}` is the `momentum` or weight_decay parameter used for the optimizer during training.                                                                   | YES          | `V1.0`            |
| `_{WU}WU`                          | `{WU}` is the number of warmup epochs if personalised via the `warmup_epochs` parameter.                                                                      | NO           | `V1.0`            |
| `_{YYYY}-{MM}-{DD}T{hh}-{mm}-{ss}` | The date and time of training.                                                                                                                                | YES          | `V1.0`            |

### *Parameters.pt*

Contains a recording of all training parameters.

### *info.txt*

Full transcript of the learning phase.

### *metrics.csv*

Comma separated log of the mean metrics and losses of the training and validation set for each epoch.

### *output.csv*

Output log on the validation set for snapshots saved during training.

| **Column names**      | **Description**                                                                                                                                                                                                                       | **Can repeat**               |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------|
| `epoch`               | Epoch number. Warning the number might be wrong by one due to `0-indexing` in arrays.                                                                                                                                                 | NO                           | 
| `id`                  | Curve ID.                                                                                                                                                                                                                             | NO                           | 
| `PRED0`, `PRED1`, ... | Raw network output values as floating point numbers.                                                                                                                                                                                  | REPEAT for each output head. | 
| `REEL0`, `REEL1`, ... | Real target output if known as floating point numbers.                                                                                                                                                                                | REPEAT for each output head. | 
| `MASK0`, `MASK1`, ... | Integer mask for output. Identifies which part of the output should be ignored when compared to `REEL` output. Does not necessarily mean that the output should be ignored in inference. Value is `0` for ignore and `1` for include. | REPEAT for each output head. | 

`PRED`, `REEL` and `MASK` vectors will repeat for each output head of the network, each head can result in vectors of different sizes but
the `PRED`, `REEL` and `MASK` triplets corresponding to the same head will be the same length.
### *snapshots* folder

This folder contains the saved snapshots with the parameters of the trained model.
Files are named `snap_{X}.pt` for regular snapshots or `snap_be_{X}.pt` for the best snapshot, with `{X}` the epoch
number.

## Usage of `.pt`

Files with the `.pt` are generated by the [pytorch](https://pytorch.org) library.
You can load those files in python using :

``` python
import torch
parameter_dict = torch.load("path/to/Parameters.pt")
```

The resulting `parameter_dict` is a python dictionary listing all parameters as `key: value` pairs.


> If you do not have Cuda for GPU calculation you can load the network on CPU using :
> ``` python
> import torch
> parameter_dict = torch.load("path/to/Parameters.pt", map_location="cpu")
> ```


