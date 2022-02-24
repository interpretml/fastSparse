# fastSparse

This repository contains source code to our AISTATS 2022 paper: 

* [Fast Sparse Classification for Generalized Linear and Additive Models](https://arxiv.org/abs/2202.11389)

Acknowledgement: For this AISTATS camera ready code repository, we build our method based on [L0Learn](https://github.com/hazimehh/L0Learn#l0learn-fast-best-subset-selection)’s codebase, so that we could use its preprocessing steps, and the pipeline for running the full regularization path of $λ_0$ values. Therefore, the computational speedup shown in our paper solely comes from our new proposed algorithms.

We plan to build our own pipeline and further extend this work before pushing the project to CRAN. Right now you can install the project from GitHub directly. Our repository will be actively maintained, and the most updated version can be found at this current GitHub page.

If you encounter any problem with installation or usage, please don't hesitate to reach out to the following email address: jiachang.liu@duke.edu.

---
## 1. Installation ##

1.1 New environment setup. We recommend using Anaconda (preferrably in a Linux system) to set up a new R environment using the the "environment.yml" file we provide. First make sure you git clone our GitHub repo and go to the fastSparse folder. Then, in your terminal, create the new environment and activate this environment:
```
git clone https://github.com/jiachangliu/fastSparse
cd fastSparse
conda env create -f environment.yml
conda activate fastSparse_environment
```

1.2 Relevant RcppArmadillo library installation. In terminal, type R. Then, inside R, do the following commands "install.packages("RcppArmadillo")". Quit R by typing "quit()". The three commands together in the terminal are
```
R
install.packages("RcppArmadillo")
quit()
```

1.3 Install our FastSparse library by typing in the terminal:
<!-- ```
R CMD INSTALL FastSparse_1.0.tar.gz
``` -->

```
R CMD build --no-build-vignettes FastSparse
R CMD INSTALL FastSparse_0.1.0.tar.gz
```

1.4 Now you can open the rstudio under this conda environment to start exploring our library by typing in the terminal
```
rstudio
```

---
## 2. Application and Usage
We provide a toolkit for producing sparse and interpretable generalized linear and additive models for the binary classiciation task by solving the L0-regularized problems. The classiciation loss can be either the logistic loss or the exponential loss. The algorithms can produce high quality (swap 1-OPT) solutions and are generally 2 to 5 times faster than previous approaches.

### 2.1 Logistic Regression
For fast sparse logistic regression, we propose to use linear/quadratic surrogate cuts that allow us to efficiently screen features for elimination, as well as use of a priority queue that favors a more uniform exploration of features.

If you inside FastSparse_0.1.0.tar.gz, the proposed linear/quadratic surrogate cuts and priority queue techniques can be found in "src/CDL012LogisticSwaps.cpp".

To fit a single pair (&lambda;0=3.0, &lambda;2=0.001) regularization and extract the coefficients, you can use the following code in your Rscript:
```
library(FastSparse)
fit <- FastSparse.fit(X_train, y_train, loss="Logistic", algorithm="CDPSI", penalty="L0L2", autoLambda=FALSE, lambdaGrid=list(3.0), nGamma=1, gammaMin=0.001, gammaMax=0.001)
beta = as.vector(coef(fit, lambda=3.0, gamma=0.001)) # first element is intercept
```

To fit a full regularization path with just a single (&lambda;2=0.001) regularization (the algorithm will automatically pick appropriate &lambda;0 values) and extract all coefficients along this regularization path, you can use the following code in your Rscript:
```
library(FastSparse)
fit <- FastSparse.fit(X_train, y_train, loss="Logistic", algorithm="CDPSI", penalty="L0L2", nGamma=1, gammaMin=0.001, gammaMax=0.001)
for (i in 1:lengths(fit$lambda)){
    lamb = fit$lambda[[1]][i]
    beta = as.vector(coef(fit, lambda=lamb, gamma=0.001)) # first element is intercept
```


### 2.2 Exponential Loss
As an alterantive to the logistic loss, we propose the exponential loss, which permits an analytical solution to the line search at each iteration.

One caveat of using the exponential loss is that make sure your X_train feature matrix are binary with each entry equal only to 0 or 1. Please refer to Appendix D.4 and Figure 10-13 in our paper to see why it is necessary for the feature matrix to be binary (0 and 1) to produce visually interpretable additive models.

If you inside FastSparse_0.1.0.tar.gz, the proposed exponential loss implementations can be found in "src/include/CDL012Exponential.h", "src/include/CDL012ExponentialSwaps.h", and "src/CDL012ExponentialSwaps.cpp".

Like the logistic loss shown above, to fit a single (&lambda;0=3.0) regularization and extract the coefficients, you can use the following code in your Rscript:
```
library(FastSparse)
fit <- FastSparse.fit(X_train, y_train, loss="Exponential", algorithm="CDPSI", penalty="L0L2", autoLambda=FALSE, lambdaGrid=list(3.0), nGamma=1, gammaMin=0.001, gammaMax=0.001)
beta = as.vector(coef(fit, lambda=3.0, gamma=0.001)) # first element is intercept
```

To fit a full regularization path (the algorithm will automatically pick appropriate &lambda;0 values) and extract all coefficients along this regularization path, you can use the following code in your Rscript:
```
library(FastSparse)
fit <- FastSparse.fit(X_train, y_train, loss="Exponential", algorithm="CDPSI", penalty="L0L2", nGamma=1, gammaMin=0.00001, gammaMax=0.001)
for (i in 1:lengths(fit$lambda)){
    lamb = fit$lambda[[1]][i]
    beta = as.vector(coef(fit, lambda=lamb, gamma=0.001)) # first element is intercept
```

Note that for the above two examples, the internal code actually does not impose &lambda;2 regularization for the exponential loss (please refer to Section 4 in our paper for the detailed reason). The "gamma=0.00001" only serves as a placeholder so that we can extract the coefficient correctly.

### 2.3 Linear Regression
Although our method is designed for classification problems, our proposed dynamic ordering technique can also speed up the local swap process for linear regression.

If you inside FastSparse_0.1.0.tar.gz, the proposed priority queue technique is implemented in "src/include/CDL012Swaps".

To fit a full regularization path with just a single (&lambda;2=0.001) regularization (the algorithm will automatically pick appropriate &lambda;0 values) and extract all coefficients along this regularization path, you can use the following code in your Rscript:
```
fit <- FastSparse.fit(X_train, y_train, penalty="L0L2", algorithm="CDPSI", maxSuppSize = 300, autoLambda=False, nGamma = 1, gammaMin = 0.001, gammaMax = 0.001)
for (i in 1:lengths(fit$lambda)){
    lamb = fit$lambda[[1]][i]
    beta = as.vector(coef(fit, lambda=lamb, gamma=0.001)) # first element is intercept
```

## 3. Experiment Replication
To replicate our experimental results, please go the following [folder](./experiments). The folder is currently under construction, and experimental scripts are being cleaned; please check back in a few days!

<!-- ## Citing Our Work ##
If you find our work useful in your research, please consider citing the following paper:

```
} -->