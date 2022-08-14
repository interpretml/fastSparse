# fastSparse

This repository contains source code to our AISTATS 2022 paper: 

* [Fast Sparse Classification for Generalized Linear and Additive Models](https://arxiv.org/abs/2202.11389)

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

