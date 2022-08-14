source("comp_time.R")

# Figure 3
dataset = "fico" # "compas"
data_type = "real"
binary = TRUE
folds = c(0, 1, 2, 3, 4)
algorithms = c("fastsparse", "l0learn")
LogFile = paste("results/time/", dataset, ".txt", sep="")
for (fold in folds){ 
  for (algorithm in algorithms){
    print(algorithm)
    lambs = c(0.8, 1, 2, 3, 4, 5, 6, 7)
    gammas = c(0.00001, 0.001)
    penalty_type = "L0L2"
    ell = "Logistic"
    comp_time(dataset, binary, data_type, LogFile, algorithm, ell, fold, penalty_type, lambs, gammas)
    if (algorithm == "fastsparse"){
      ell = "Exponential"
      comp_time(dataset, binary, data_type, LogFile, algorithm, ell, fold, penalty_type, lambs, gammas)
    }
  }
}
