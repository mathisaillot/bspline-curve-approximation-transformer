RDG README File Template --- General --- Version: 0.1 (2022-10-04)

This README file was generated on [2025-03-28] by [Mathis Saillot].
Last updated: [2025-04-10].

# GENERAL INFORMATION

## Dataset title: B-spline Dataset of *B-spline Curve Approximation With Transformer Neural Networks* article

## DOI: [doi:10.57745/O7RCHI](https://doi.org/10.57745/O7RCHI)

### Contact email: mathis.saillot@univ-lorraine.fr

# METHODOLOGICAL INFORMATION

This dataset contains Cubic B-spline curves with randomly generated knots and control points.

# DATA & FILE OVERVIEW

## Information files:

### [dataset_info.csv](dataset_info.csv)

It contains the list of `.csv` files containing the data. Those files are located inside the [curves](curves) folder.

| **Column Name**  | **Description**                                                      |
|------------------|----------------------------------------------------------------------|
| `File Name`      | Name of the file containing the data.                                |
| `Internal Knots` | Number of internal Knots of the B-splines in the corresponding file. |
| `Order`          | Order of the B-spline. Will always be `4` here.                      |
| `First Id`       | Id of the first curve in the file.                                   |
| `Count`          | Number of curves in the file.                                        |

### [train_idx.txt](train_idx.txt)

It contains the list of `150,000` curve IDs of the training set, in increasing order of IDs.

### [train_idx.txt](train_idx.txt)

It contains the list of `10,000` curve IDs of the validation set, in increasing order of IDs.

## Files in [curves](curves) folder

### Naming convention

Files are named `bsplineparam_kX_mY_idZ.csv` with :

- `X` the order of the b-spline.
- `Y` the number of internal knots.
- `Z` the id of the first curve in the file.

### Data description

The csv files contain one b-spline per line, with comma separated floating points values.
The exact number of data points depends on the Order and the number of knots of the Bspline.

| **Column Name**                 | **Description**                                             |
|---------------------------------|-------------------------------------------------------------|
| `Id`                            | Id of the curve.                                            |
| `U0`, `U1`, ...                 | Knots of the Bspline. With k repeating knots at end-points. |
| `PX0`, `PY0`, `PX1`, `PY1`, ... | 2D coordinates of the control points of the Bspline.        |

All floating points numbers use `.` separators.

## Related links

Code is available on github that uses this dataset for neural network training : 
[bspline-curve-approximation-transformer](https://github.com/mathisaillot/bspline-curve-approximation-transformer).

Linked [pretrained neural network dataset](https://doi.org/10.57745/IQ91KC).

Based on this article : ["B-spline Curve Approximation With Transformer Neural Networks"](https://hal.science/hal-04931216v1).
