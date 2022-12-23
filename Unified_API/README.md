# Table of Content <!-- omit in toc -->
- [Overview](#overview)
  - [Comparison between FastSparse and FasterRisk](#comparison-between-fastsparse-and-fasterrisk)
  - [Loss functions for different models:](#loss-functions-for-different-models)
- [API - Sparse Additive Models](#api---sparse-additive-models)
  - [FastSparse:](#fastsparse)
    - [1. Fit on Data](#1-fit-on-data)
      - [1a. Fit with No Monotonicity Constraint](#1a-fit-with-no-monotonicity-constraint)
      - [1b. Fit with Monotonicity Constraint](#1b-fit-with-monotonicity-constraint)
    - [2. Get Parameters after Fitting](#2-get-parameters-after-fitting)
    - [3. Predict](#3-predict)
  - [FasterRisk](#fasterrisk)
    - [1. Fit on Data](#1-fit-on-data-1)
    - [2. Get Parameters after Fitting](#2-get-parameters-after-fitting-1)
    - [3. Predict](#3-predict-1)

# Overview 
The end goal is to provide a unified API so that people can use both FastSparse and FasterRisk from the same library. To do this, the easiest way is to write another python library wrapper on top of the FastSparse and FasterRisk python libraries. We call this library **Sparse Additive Models**.

When we use pip to install sparseadditivemodels, pip will first install fastsparse and fasterrisk. When we use sparseadditivemodels's functions, the function will call the corresponding function from FastSparse or FasterRisk.

FasterRisk is already in python implementation and published on PyPI. The main enigieering difficulty is to **build a python library** for FastSparse (with C++ Armadillo implementation) without calling R.

## Comparison between FastSparse and FasterRisk
Summary of differences between FastSparse and FasterRisk:
| Key Differences      | FastSparse | FasterRisk     |
| :---        |    :----   |         :--- |
| sparsity is controlled      | implicitly by $\ell_0$ regularization       | explicitly by support size $k$   |
|objective function| $\mathcal{L}(\beta) = \sum_i^n l(y_i, x_i, \beta) + \lambda_2 \Vert \beta \Vert _2^2 + \lambda_0 \Vert \beta \Vert _0$ | <p> $\mathcal{L}(\beta) = \sum_i^n l(y_i, x_i, \beta)$ <p> subject to $\Vert \beta \Vert _0 \leq k$ |
| model options   | <p> Linear Regression <p> Logistic Regression <p> Adaboost (Note: $\lambda_2$ must be $0$) | Logistic Regression      |
| continuous coefficient | &check; | &cross; (not yet; can be done easily) |
| integer coefficient | &cross; | &check; |
| bound on coefficient| &cross; (not yet; can be done with effort) | &check; |


## Loss functions for different models:

| model | loss name | loss function $l(\cdot)$ |
| :-- | :-- | :-- |
|Linear Regression | square loss | $l(y_i, x_i, \beta) := (y_i - x_i^T \beta)^2$|
|Logistic Regression | logistic loss| $l(y_i, x_i, \beta) := \log(1+\exp(-y_i x_i^T \beta))$|
|Adaboost | exponential loss |$l(y_i, x_i, \beta) := \exp(-y_i x_i^T \beta)$|


# API - Sparse Additive Models
We call our unified API "SparseAdditiveModels", which can be used to call either FastSparse or FasterRisk.

```python
import SparseAdditiveModels as SAM
```

## FastSparse:
### 1. Fit on Data
#### 1a. Fit with No Monotonicity Constraint
Fit with a specified $(\lambda_2, \lambda_0)$ pair.
```python
# X_trian.shape = (n, p), y_train.shape = (n, ), lambda2 and lambda0 are scalars
sam = SAM.FastSparse_fit(X=X_train, y=y_train, loss="Square", lambda2=lambda2, lambda0=lambda0) # loss can also be "Logistic" or "Exponential" 
```

Fit with a specified $\lambda_2$ value and maximum support size $k$. The algorithm will fit a regularization path with different $\lambda_0$ values (chosen automatically) from large to small until the support size exceeds $k$.
```python
# X_trian.shape = (n, p), y_train.shape = (n, ), lambda2 and k are scalars
sam = SAM.FastSparse_fit_path(X=X_train, y=y_train, loss="Square", lambda2=lambda2, maxSupp=k) # loss can also be "Logistic" or "Exponential"
```

#### 1b. Fit with Monotonicity Constraint
To impose monotonicity constraint, we can achieve this by adding lower and upper bounds on the coefficients. For example, to constrain a coefficient to be positive, we set the lower bound to be 0, and the upper bound to be a very large number like $1 \times 10^6$.

Fit with a specified $(\lambda_2, \lambda_0)$ pair and lower and upper bounds on each coefficient.
```python
# X_trian.shape = (n, p), y_train.shape = (n, ), lambda2 and lambda0 are scalars, coeff_low.shape = (p, ), coeff_high.shape = (p, )
sam = SAM.FastSparse_fit(X=X_train, y=y_train, loss="Square", lambda2=lambda2, lambda0=lambda0, coeffs_low=coeff_low, coeff_high=coeff_high)
```

Fit with a specified $\lambda_2$ value, maximum support size $k$, and lower and upper bounds on each coefficient. The algorithm will fit a regularization path with different $\lambda_0$ values (chosen automatically) from large to small until the support size exceeds $k$.
```python
# X_trian.shape = (n, p), y_train.shape = (n, ), lambda2 and k are scalars, coeff_low.shape = (p, ), coeff_high.shape = (p, )
sam = SAM.FastSparse_fit_path(X=X_train, y=y_train, loss="Square", lambda2=lambda2, maxSupp=k, coeff_low=coeff_low, coeff_high=coeff_high) # loss can also be "Logistic" or "Exponential"
```

### 2. Get Parameters after Fitting
```python
print("training time is {}".format(sam.train_duration))
for model in sam.models :
        # coeff.shape = (p, ), intercept, lambda2 and lambda0 are scalars
        coeff, intercept, lambda2, lambda0 = model.coeff_ model.intercept, lambda2, lambda0
        print("The coefficients are {}; the intercept is {}; lambda2 is {}; lambda0 is {}".format(coeff, intercept, lambda2, lambda0))
```

### 3. Predict
```python
for model in sam.models:
        # X_test.shape = (n_test, p), y_test.shape = (n_test, )
        y_test = model.predict(X=X_test)
```

## FasterRisk
### 1. Fit on Data
Fit with a specified support size $k$, a coefficient lower bound coeff$_{low}$, and a coefficient upper bound coeff$_{high}$ with **continuous** coefficients
```python
# X_train.shape = (n, p), y_train.shape = (n, ), k is a scalar, coeff_low.shape = (p, ), coeff_high.shape = (p, )
sam = SAM.FastSparse_fit(X=X_train, y=y_train, suppSize=k, coeff_low=coeff_low, coeff_high=coeff_high, coefficient_type="continuous")
```

Fit with a specified support size $k$, a coefficient lower bound coeff$_{low}$, and a coefficient upper bound coeff$_{high}$ with **integer** coefficients
```python
# X_train.shape = (n, p), y_train.shape = (n, ), k is a scalar, coeff_low.shape = (p, ), coeff_high.shape = (p, )
sam = SAM.FastSparse_fit(X=X_train, y=y_train, suppSize=k, coeff_low=coeff_low, coeff_high=coeff_high, coefficient_type="integer")
```



### 2. Get Parameters after Fitting
```python
print("training time is {}".format(sam.train_duration))
for model in sam.models :
        # coeff.shape = (p, ), intercept and multiplier are scalars
        coeff, intercept, multiplier = model.coeff_ model.intercept, lambda2, lambda0
        print("The coefficients are {}; the intercept is {}; the multiplier is {}".format(coeff, intercept, multiplier))
```

### 3. Predict
```python
for model in sam.models:
        # X_test.shape = (n_test, p), y_test.shape = (n_test, )
        y_test = model.predict(X=X_test)
```