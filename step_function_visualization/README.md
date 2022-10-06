# fastSparse

This page contains the demo code to plot the step functions shown in our AISTATS 2022 paper: 

* [Fast Sparse Classification for Generalized Linear and Additive Models](https://arxiv.org/abs/2202.11389)

To plot sparse step functions, we need to do the following two steps:


## 1. Binarization Preprocessing
Binarize all continuous features into {0, 1} binary features by creating thresholds. This is a preprocessing step. The binarization helper function is in [binarization_utils.py](./binarization_utils.py)

An example is given in the [binarize_continuousData_demo notebook](./binarize_continuousData_demo.ipynb).

## 2. Step Function Plotting
Apply fastSparse on the preprocessed binary features to produce sparse coefficients and apply the plot functions in [plot_utils.py](./plot_utils.py). 

An example is given in the [plot_stepFunction_demo notebook](./plot_stepFunction_demo.ipynb), which reproduces the FICO step functions shown in the paper. Or you can modify the notebook to tailor to plot step functions on your own data.

