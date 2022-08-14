# fastSparse

This page contains the source code to reproduce results in our AISTATS 2022 paper: 

* [Fast Sparse Classification for Generalized Linear and Additive Models](https://arxiv.org/abs/2202.11389)

The FICO dataset is publically available. You can request the data from this [link](https://community.fico.com/s/explainable-machine-learning-challenge).

To convert the original FICO dataset with continuous features into binary features, please refer to our header in this [file](../visualization/fico_bin_first_5_rows.csv).

## 1. Time Comparison Experiment
For the time comparison experiments, please run the following line in your terminal

```
Rscript run_time.R
```

## 2. Solution Quality Experiment of the Entire Regularization Path
For the solution quality experiment, please run the following line in your terminal

```
Rscript run_baseline.R
```
The above script contains code to run experiments on both the real datasets and the synthetically generated data.