library(reticulate)
library(MASS)
  
dataset = "high_corr"
sim_seeds = c(1,2,3,4,5)
sim_p = 1000
sim_n = 800
sim_k = 25
np = import("numpy") 

for (sim_seed in sim_seeds){
  set.seed(sim_seed)
  mat = matrix(rep(c(0:(sim_p-1)), sim_p), nrow=sim_p, ncol=sim_p, byrow=TRUE)
  Sigma = .9 ** abs(mat - t(mat))
    
  X = mvrnorm(n=1.2*sim_n, rep(0, sim_p), Sigma)
  B = c(rep(0, sim_p))
  step = floor(sim_p/sim_k)
  for (l in 1:sim_k){
    B[l*step] = 1
  }
  prob = 1/(1+exp(-1*(X %*% B))) # check prob
  y = rep(1, sim_n*1.2)
  y[prob < 0.5] = -1

  X_test = X[(sim_n+1):(sim_n*1.2), 1:sim_p]
  y_test = y[(sim_n+1):(sim_n*1.2)]
  test = cbind(X_test, y_test)

  X_train = X[1:sim_n, 1:sim_p]
  y_train = y[1:sim_n]
  train = cbind(X_train, y_train)

  trainfile = paste("datasets/high_corr/high_corr_train", sim_n, sim_p, sim_k, sim_seed, sep="_")
  trainfile = paste(trainfile, ".npy", sep="")
  np$save(trainfile, train)

  Bfile = paste("datasets/high_corr/high_corr_coef", sim_n, sim_p, sim_k, sim_seed, sep="_")
  np$save(paste(Bfile, ".npy", sep=""), B)

  testfile = paste("datasets/high_corr/high_corr_test", sim_n, sim_p, sim_k, sim_seed, sep="_")
  testfile = paste(testfile, ".npy", sep="")
  np$save(testfile, test)
}