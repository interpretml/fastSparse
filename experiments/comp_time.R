source("utils.R")
library(L0Learn)
library(FastSparse)
library(dplyr)
library(reticulate)

fs = function(dataset, data_type, penalty_type, lambs, gammas, X_train, y_train, 
                   X_test, y_test, LogFile, B, sim_seed, binary, fold, ell){
  print(paste("train model", penalty_type, sep=" "))
  n = dim(X_train)[1]
  p = dim(X_train)[2]
  if (penalty_type != "L0") {
    gammas = gammas
  } else {
    gammas = c(0)
  }
  for (g in gammas){
    for (l in lambs){
      start_time <- Sys.time()
      fit <- FastSparse.fit(X_train, y_train, loss=ell, algorithm="CDPSI",
                         penalty=penalty_type, autoLambda=FALSE, lambdaGrid=list(l), nGamma=1, gammaMin=g, gammaMax=g)
      end_time = Sys.time()
      print("train finished")
      train_duration = difftime(end_time, start_time, units="secs")
      print(train_duration)
      for (i in 1:lengths(fit$lambda)){
        lamb = fit$lambda[[1]][i]
        beta = as.vector(coef(fit, lambda=lamb, gamma=g)) # first element is intercept
        if (length(beta) != p+1){
          if (data_type == "real"){
            out= c(dataset, data_type, binary, fold, n, p, paste("fastsparse", ell), penalty_type, lamb, g, NA,
                  fit$suppSize[[1]][i], NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, train_duration, "\n")
          } else{
            out= c(dataset, data_type, binary, fold, n, p, paste("fastsparse", ell), penalty_type, lamb, g, sim_seed,
                  fit$suppSize[[1]][i], NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, train_duration, "\n")
          }
          write_log(LogFile, out)
          next
        } else {
          if (ell=="Exponential"){
            intercept = beta[1]
            b = beta[2:length(beta)]

            pred_train =  rep(1, dim(X_train)[1])
            f_train = X_train %*% b + intercept
            pred_train[f_train < 0] = -1
            prob_train = exp(2*f_train) / (1+exp(2*f_train))

            
            pred_test =  rep(1, dim(X_test)[1])
            f_test = X_test %*% b + intercept
            pred_test[f_test < 0] = -1
            prob_test = exp(2*f_test) / (1+exp(2*f_test))

            acc_train = 1 - (sum(y_train != pred_train)/length(y_train))
            acc_test = 1 - (sum(y_test != pred_test)/length(y_test))

            exp_loss_train = sum(exp(-y_train * f_train))
            exp_loss_test = sum(exp(-y_test * f_test))

            log_loss_train = get_logistic_loss(intercept, b, X_train, y_train)
            log_loss_test = get_logistic_loss(intercept, b, X_test, y_test)
            if (fit$penalty == "L0L2"){
              penalty_term = lamb * fit$suppSize[[1]][i] + g * sum(beta[beta != 0]^2)
            } else if (fit$penalty == "L0L1") {
              penalty_term = lamb * fit$suppSize[[1]][i] + g * sum(abs(beta[beta != 0]))
            } else {
              penalty_term = lamb * fit$suppSize[[1]][i]
            }
          
            obj_train = log_loss_train + penalty_term
            obj_test = log_loss_test + penalty_term
            if (data_type == "real"){
              out= c(dataset, data_type, binary, fold, n, p, paste("fastsparse", ell), penalty_type, lamb, g, NA, 
                      fit$suppSize[[1]][i], get_auc(y_train, prob_train), get_auc(y_test, prob_test), 
                      acc_train, acc_test, log_loss_train, log_loss_test, obj_train, obj_test, NA, NA, NA,
                      exp_loss_train, exp_loss_test, train_duration, "\n")
            } else{
              print("check dataset!")
            }
            write_log(LogFile, out)
            # cat(c(intercept, b[b!= 0]), file=paste(LogFile, paste("fastsparse", ell), fold, lamb, g, "coeff", sep="_"), append=TRUE, sep=";")
            # cat(which(b!=0), file=paste(LogFile, paste("fastsparse", ell), fold, lamb, g, "index", sep="_"), append=TRUE, sep=";")
          } else{
            intercept = beta[1]
            b = beta[2:length(beta)]  
            pred_train = predict(fit, newx=X_train, lambda=lamb, gamma=g)
            pred_test = predict(fit, newx=X_test, lambda=lamb, gamma=g)

            normCenteredX = get_norm_from_centeredX(X_train)

            if (fit$penalty == "L0L2"){
              penalty_term = lamb * fit$suppSize[[1]][i] + g * sum((b[b != 0]*normCenteredX[b != 0])^2)
            } else if (fit$penalty == "L0L1") {
              penalty_term = lamb * fit$suppSize[[1]][i] + g * sum(abs(b[b != 0]*normCenteredX[b != 0]))
            } else {
              penalty_term = lamb * fit$suppSize[[1]][i]
            }
          
            results = get_results(beta, pred_train, pred_test, X_train, y_train, X_test, y_test, 
                        data_type, B, penalty_term, train_duration)
          
            param = c(dataset, data_type, binary, fold, n, p, paste("fastsparse", ell), penalty_type, lamb, g)
            out = append(param, results)
            write_log(LogFile, out)
            # cat(c(intercept, b[b!= 0]), file=paste(LogFile, paste("fastsparse", ell), fold, lamb, g, "coeff", sep="_"), append=TRUE, sep=";")
            # cat(which(b!=0), file=paste(LogFile, paste("fastsparse", ell), fold, lamb, g, "index", sep="_"), append=TRUE, sep=";")
          }
          
        }
        
      }
    }
  }
}



l0learn = function(dataset, data_type, penalty_type, lambs, gammas, X_train, y_train, 
                   X_test, y_test, LogFile, B, sim_seed, binary, fold){
  print(paste("train model", penalty_type, sep=" "))
  n = dim(X_train)[1]
  p = dim(X_train)[2]
  if (penalty_type != "L0") {
    gammas = gammas
  } else {
    gammas = c(0)
  }
  for (g in gammas){
    for (l in lambs){
      start_time <- Sys.time()
      fit <- L0Learn.fit(X_train, y_train, loss="Logistic", algorithm="CDPSI",
                       penalty=penalty_type, autoLambda=FALSE, lambdaGrid=list(l), nGamma=1, gammaMin=g, gammaMax=g)
      end_time = Sys.time()
      
      print("train finished")
      train_duration = difftime(end_time, start_time, units="secs")
      print(train_duration)
      for (i in 1:lengths(fit$lambda)){
        lamb = fit$lambda[[1]][i]
        beta = as.vector(coef(fit, lambda=lamb, gamma=g)) # first element is intercept
        if (length(beta) != p+1){
          if (data_type == "real"){
            out= c(dataset, data_type, binary, fold, n, p, "l0learn", penalty_type, lamb, g, NA,
                  fit$suppSize[[1]][i], NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, train_duration, "\n")
          } else{
            out= c(dataset, data_type, binary, fold, n, p, "l0learn", penalty_type, lamb, g, sim_seed,
                  fit$suppSize[[1]][i], NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, train_duration, "\n")
          }
          write_log(LogFile, out)
          next
        } else {
          intercept = beta[1]
          b = beta[2:length(beta)]  
          pred_train = predict(fit, newx=X_train, lambda=lamb, gamma=g)
          pred_test = predict(fit, newx=X_test, lambda=lamb, gamma=g)

          normCenteredX = get_norm_from_centeredX(X_train)

          if (fit$penalty == "L0L2"){
            penalty_term = lamb * fit$suppSize[[1]][i] + g * sum((b[b != 0]*normCenteredX[b != 0])^2)
          } else if (fit$penalty == "L0L1") {
            penalty_term = lamb * fit$suppSize[[1]][i] + g * sum(abs(b[b != 0]*normCenteredX[b != 0]))
          } else {
            penalty_term = lamb * fit$suppSize[[1]][i]
          }
          
          results = get_results(beta, pred_train, pred_test, X_train, y_train, X_test, y_test, 
                        data_type, B, penalty_term, train_duration)
          
          param = c(dataset, data_type, binary, fold, n, p, "l0learn", penalty_type, lamb, g)
          out = append(param, results)
          write_log(LogFile, out)
          # cat(c(intercept, b[b!= 0]), file=paste(LogFile, "l0learn", fold, lamb, g, "coeff", sep="_"), append=TRUE, sep=";")
          # cat(which(b!=0), file=paste(LogFile, "l0learn", fold, lamb, g, "index", sep="_"), append=TRUE, sep=";")
        }
        
      }
    }

  }
}



comp_time = function(dataset, binary, data_type, LogFile, algorithm, ell=NULL, fold=NULL,
                penalty_type=NULL, lambs=NULL, gammas=NULL,
                sim_seed=NULL, sim_n=NULL, sim_p=NULL, sim_k=NULL, g=NULL){
  
  data_info = get_dataset(dataset, binary, data_type, fold, 
                      sim_seed, sim_n, sim_p, sim_k)
  X_train = data_info$X_train
  y_train = data_info$y_train
  X_test = data_info$X_test
  y_test = data_info$y_test
  B = data_info$B
  
  if (algorithm == "fastsparse"){
    fs(dataset, data_type, penalty_type, lambs, gammas, 
          X_train, y_train, X_test, y_test, LogFile, B, sim_seed, binary, fold, ell)
  } else if (algorithm == "l0learn"){
    l0learn(dataset, data_type, penalty_type, lambs, gammas, 
          X_train, y_train, X_test, y_test, LogFile, B, sim_seed, binary, fold)
  } else {
    print("check algorithm")
  }
}



