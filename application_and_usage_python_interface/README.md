# fastSparse

This repository contains source code to our AISTATS 2022 paper: 

* [Fast Sparse Classification for Generalized Linear and Additive Models](https://arxiv.org/abs/2202.11389)

Update (09/17/23): We have created a new python package called [FastSparseGAMS](https://github.com/ubc-systopia/L0Learn/tree/master/python). Instead of using a python wrapper as stated below, you can now install FastSparse via pip directly through the following commands:

```bash
pip install fastsparsegams
```

Please go to FastSparseGAM's [tutorial page](https://github.com/ubc-systopia/L0Learn/blob/master/python/tutorial_example/example.py) for examples on how to use the new interface.

The instructions (Section 2) below still work, but we recommand using the new python package first.

## 2. Application and Usage - Python Interface
**We provide a wrapper to use fastSparse in a python environment**

**To use fastSparse directly in an R environment, please go to the folder [application_and_usage_R_interface](../application_and_usage_R_interface).**

We provide a toolkit for producing sparse and interpretable generalized linear and additive models for the binary classiciation task by solving the L0-regularized problems. The classiciation loss can be either the logistic loss or the exponential loss. The algorithms can produce high quality (swap 1-OPT) solutions and are generally 2 to 5 times faster than previous approaches.

### 2.0 Import the python wrapper
We need to use a python wrapper to interact with the R code. To do this, make sure to copy and paste the following code at the top of your python file

```python
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
base = importr('base')
d = {'package.dependencies': 'package_dot_dependencies'}
FastSparse = importr('FastSparse', robject_translations = d)
```

### 2.1 Logistic Regression
For fast sparse logistic regression, we propose to use linear/quadratic surrogate cuts that allow us to efficiently screen features for elimination, as well as use of a priority queue that favors a more uniform exploration of features.

If you go inside FastSparse_0.1.0.tar.gz, the proposed linear/quadratic surrogate cuts and priority queue techniques can be found in "src/CDL012LogisticSwaps.cpp".

To fit a single pair (&lambda;0=3.0, &lambda;2=0.001) regularization and extract the coefficients, you can use the following code in your Rscript:
```python
fit = FastSparse.FastSparse_fit(X_train, y_train, loss="Logistic", algorithm="CDPSI", penalty="L0L2", autoLambda=FALSE, lambdaGrid=[3.0], nGamma=1, gammaMin=0.001, gammaMax=0.001)
beta = np.asarray(base.as_matrix(FastSparse.coef_FastSparse(fit))) # first element is intercept
```

To fit a full regularization path with just a single (&lambda;2=0.001) regularization (the algorithm will automatically pick appropriate &lambda;0 values) and extract all coefficients along this regularization path, you can use the following code in your Rscript:
```python
fit = FastSparse.FastSparse_fit(X_train, y_train, loss="Logistic", algorithm="CDPSI", penalty="L0L2", nGamma=1, gammaMin=0.001, gammaMax=0.001)

lambda0s = np.asarray(base.as_matrix(fit.rx2('lambda')))[0]
betas = np.asarray(base.as_matrix(FastSparse.coef_FastSparse(fit)))

# examine pairs of lambda0, beta
for i in range(len(lambda0s)):
    lambda0 = lambda0s[i]
    beta = betas[:, i] # first element is intercept
```


### 2.2 Exponential Loss
As an alterantive to the logistic loss, we propose the exponential loss, which permits an analytical solution to the line search at each iteration.

One caveat of using the exponential loss is that make sure your X_train feature matrix are binary with each entry equal only to 0 or 1. Please refer to Appendix D.4 and Figure 10-13 in our paper to see why it is necessary for the feature matrix to be binary (0 and 1) to produce visually interpretable additive models.

If you inside FastSparse_0.1.0.tar.gz, the proposed exponential loss implementations can be found in "src/include/CDL012Exponential.h", "src/include/CDL012ExponentialSwaps.h", and "src/CDL012ExponentialSwaps.cpp".

Like the logistic loss shown above, to fit a single (&lambda;0=3.0) regularization and extract the coefficients, you can use the following code in your Rscript:
```python
fit = FastSparse.FastSparse_fit(X_train, y_train, loss="Exponential", algorithm="CDPSI", penalty="L0L2", autoLambda=FALSE, lambdaGrid=[3.0], nGamma=1, gammaMin=0.001, gammaMax=0.001)
beta = np.asarray(base.as_matrix(FastSparse.coef_FastSparse(fit))) # first element is intercept
```

To fit a full regularization path (the algorithm will automatically pick appropriate &lambda;0 values) and extract all coefficients along this regularization path, you can use the following code in your Rscript:
```python
fit = FastSparse.FastSparse_fit(X_train, y_train, loss="Exponential", algorithm="CDPSI", penalty="L0L2", nGamma=1, gammaMin=0.00001, gammaMax=0.00001)

lambda0s = np.asarray(base.as_matrix(fit.rx2('lambda')))[0]
betas = np.asarray(base.as_matrix(FastSparse.coef_FastSparse(fit)))

# examine pairs of lambda0, beta
for i in range(len(lambda0s)):
    lambda0 = lambda0s[i]
    beta = betas[:, i] # first element is intercept
```

Note that for the above two examples, the internal code actually does not impose &lambda;2 regularization for the exponential loss (please refer to Section 4 in our paper for the detailed reason). The "gamma=0.00001" only serves as a placeholder so that we can extract the coefficient correctly.

### 2.3 Linear Regression
Although our method is designed for classification problems, our proposed dynamic ordering technique can also speed up the local swap process for linear regression.

If you inside FastSparse_0.1.0.tar.gz, the proposed priority queue technique is implemented in "src/include/CDL012Swaps".

To fit a full regularization path with just a single (&lambda;2=0.001) regularization (the algorithm will automatically pick appropriate &lambda;0 values) and extract all coefficients along this regularization path, you can use the following code in your Rscript:
```python
fit <- FastSparse.fit(X_train, y_train, penalty="L0L2", algorithm="CDPSI", maxSuppSize = 300, autoLambda=False, nGamma = 1, gammaMin = 0.001, gammaMax = 0.001)
for (i in 1:lengths(fit$lambda)){
    lamb = fit$lambda[[1]][i]
    beta = as.vector(coef(fit, lambda=lamb, gamma=0.001)) # first element is intercept

fit = FastSparse.FastSparse_fit(X_train, y_train, algorithm="CDPSI", penalty="L0L2", nGamma=1, gammaMin=0.001, gammaMax=0.001)

lambda0s = np.asarray(base.as_matrix(fit.rx2('lambda')))[0]
betas = np.asarray(base.as_matrix(FastSparse.coef_FastSparse(fit)))

# examine pairs of lambda0, beta
for i in range(len(lambda0s)):
    lambda0 = lambda0s[i]
    beta = betas[:, i] # first element is intercept
```
