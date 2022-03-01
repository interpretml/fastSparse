source("comp_baselines.R")

# Figure 5
# run real dataset
dataset = "compas"
data_type = "real"
binary = TRUE
folds = c(0, 1, 2, 3, 4)

algorithms = c("fastsparse", "l0learn", "lasso", "MCP")
LogFile = paste("results/baselines/", dataset, ".txt", sep="")
for (fold in folds){ 
  for (algorithm in algorithms){
    print(algorithm)
    gammas = c(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10)
    penalty_type = "L0L2"
    if (algorithm == "fastsparse"){
      for (ell in c("Logistic", "Exponential")){
        comp_mb_perf(dataset, binary, data_type, LogFile, algorithm, ell, fold, penalty_type, gammas)
      }
    } else {
      ell = NULL
      comp_mb_perf(dataset, binary, data_type, LogFile, algorithm, ell, fold, penalty_type, gammas)
    }
  }
}


# Figure 4
# run simulated dataset 
dataset = "high_corr"
data_type = "sim"

binary = FALSE
fold = NULL
seeds = c(1,2,3,4,5) 
n = 800
p = 1000
k = 25

algorithms = c("fastsparse", "l0learn", "lasso", "MCP")
LogFile = paste("results/baselines/", dataset, ".txt", sep="")
for (seed in seeds){ 
  for (algorithm in algorithms){
    print(algorithm)
    gammas = c(1e-9, 1e-7, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10)
    penalty_type = "L0L2"
    if (algorithm == "fastsparse"){
      ell = "Logistic"
    } else {
      ell = NULL
    }
    comp_mb_perf(dataset, binary, data_type, LogFile, algorithm, ell, fold, penalty_type, gammas,
                  sim_seed = seed, sim_n=n, sim_p=p, sim_k=k)
  }
}
