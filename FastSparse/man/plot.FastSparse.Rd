% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/plot.R
\name{plot.FastSparse}
\alias{plot.FastSparse}
\title{Plot Regularization Path}
\usage{
\method{plot}{FastSparse}(x, gamma = 0, showLines = FALSE, ...)
}
\arguments{
\item{x}{The output of FastSparse.fit}

\item{gamma}{The value of gamma at which to plot.}

\item{showLines}{If TRUE, the lines connecting the points in the plot are shown.}

\item{...}{ignore}
}
\description{
Plots the regularization path for a given gamma.
}
\examples{
# Generate synthetic data for this example
data <- GenSynthetic(n=500,p=1000,k=10,seed=1)
X = data$X
y = data$y
# Fit an L0 Model with a maximum of 50 non-zeros
fit <- FastSparse.fit(X, y, penalty="L0", maxSuppSize=50)
plot(fit, gamma=0)

}
