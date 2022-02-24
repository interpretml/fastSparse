#' @title Print FastSparse.fit object
#'
#' @description Prints a summary of FastSparse.fit
#' @param x The output of FastSparse.fit or FastSparse.cvfit
#' @param ... ignore
#' @method print FastSparse
#' @export
print.FastSparse <- function(x, ...)
{
		gammas = rep(x$gamma, times=lapply(x$lambda, length) )
		data.frame(lambda = unlist(x["lambda"]), gamma = gammas, suppSize = unlist(x["suppSize"]), row.names = NULL)
}

#' @rdname print.FastSparse
#' @method print FastSparseCV
#' @export
print.FastSparseCV <- function(x, ...)
{
    print.FastSparse(x$fit)
}
