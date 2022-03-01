source("utils.R")
library(L0Learn)
library(FastSparse)
library(dplyr)
library(glmnet)
library(ncvreg)
library(reticulate)
library(MASS)

fs = function(dataset, data_type, penalty_type, gammas, X_train, y_train, 
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
    start_time <- Sys.time()
    fit <- FastSparse.fit(X_train, y_train, loss=ell, algorithm="CDPSI",
                       penalty=penalty_type, nGamma=1, gammaMin=g, gammaMax=g)
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
          out= c(dataset, data_type, binary, NA, n, p, paste("fastsparse", ell), penalty_type, lamb, g, sim_seed,
                  fit$suppSize[[1]][i], NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, train_duration, "\n")
        }
        write_log(LogFile, out)
        next
      } else {
        if (ell=="Exponential"){
          intercept = beta[1]
          b = beta[2:length(beta)]
          # pred_train = predict(fit, newx=X_train, lambda=lamb, gamma=g)
          # pred_test = predict(fit, newx=X_test, lambda=lamb, gamma=g)

          # acc_train = get_acc(y_train, pred_train)
          # acc_test = get_acc(y_test, pred_test)

          # exp_loss_train = sum(exp(y_train*(0.5*log(pred_train/(1-pred_train)))))
          # exp_loss_test = sum(exp(y_test*(0.5*log(pred_test/(1-pred_test)))))

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
        } else{
          intercept = beta[1]
          b = beta[2:length(beta)]  
          pred_train = predict(fit, newx=X_train, lambda=lamb, gamma=g)
          pred_test = predict(fit, newx=X_test, lambda=lamb, gamma=g)

          normCenteredX = get_norm_from_centeredX(X_train)

          if (fit$penalty == "L0L2"){
            # penalty_term = lamb * fit$suppSize[[1]][i] + g * sqrt(sum(beta[beta != 0]^2))
            penalty_term = lamb * fit$suppSize[[1]][i] + g * sum((b[b != 0]*normCenteredX[b != 0])^2)
          } else if (fit$penalty == "L0L1") {
            # penalty_term = lamb * fit$suppSize[[1]][i] + g * sum(abs(beta[beta != 0]))
            penalty_term = lamb * fit$suppSize[[1]][i] + g * sum(abs(b[b != 0]*normCenteredX[b != 0]))
          } else {
            penalty_term = lamb * fit$suppSize[[1]][i]
          }
        
          results = get_results(beta, pred_train, pred_test, X_train, y_train, X_test, y_test, 
                      data_type, B, penalty_term, train_duration, sim_seed)
          if (data_type=="sim"){
            param = c(dataset, data_type, binary, NA, n, p, paste("fastsparse", ell), penalty_type, lamb, g)
          } else {
            param = c(dataset, data_type, binary, fold, n, p, paste("fastsparse", ell), penalty_type, lamb, g)
          }
          out = append(param, results)
          write_log(LogFile, out)
        }
        
      }
    }
  }
}

l0learn = function(dataset, data_type, penalty_type, gammas, X_train, y_train, 
                   X_test, y_test, LogFile, B, sim_seed, binary, fold){
  print(paste("train model", penalty_type, sep=" "))
  n = dim(X_train)[1]
  p = dim(X_train)[2]
  if (penalty_type != "L0") { # check
    gammas = gammas
  } else {
    gammas = c(0)
  }
  for (g in gammas){
    start_time <- Sys.time()
    fit <- L0Learn.fit(X_train, y_train, loss="Logistic", algorithm="CDPSI",
                       penalty=penalty_type, nGamma=1, gammaMin=g, gammaMax=g)
    end_time = Sys.time()
    print("train finished")
    train_duration = difftime(end_time, start_time, units="secs")
    print(train_duration)
    for (i in 1:lengths(fit$lambda)){
      lamb = fit$lambda[[1]][i]
      beta = as.vector(coef(fit, lambda=lamb, gamma=g)) # first element is intercept
      if (length(beta) != p+1){ # high correlation dataset
        if (data_type == "real"){
          out= c(dataset, data_type, binary, fold, n, p, "l0learn", penalty_type, lamb, g, NA,
                 fit$suppSize[[1]][i], NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, train_duration, "\n")
        } else{
          out= c(dataset, data_type, binary, NA, n, p, "l0learn", penalty_type, lamb, g, sim_seed,
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
          penalty_term = lamb * fit$suppSize[[1]][i] + g * sum((b[b != 0]*normCenteredX[b != 0])^2) # fix l2 loss, transform beta
        } else if (fit$penalty == "L0L1") {
          penalty_term = lamb * fit$suppSize[[1]][i] + g * sum(abs(beta[beta != 0])) # fix l1 loss, transform beta
        } else {
          penalty_term = lamb * fit$suppSize[[1]][i]
        }
         
        results = get_results(beta, pred_train, pred_test, X_train, y_train, X_test, y_test, 
                      data_type, B, penalty_term, train_duration, sim_seed)
        if (data_type=="sim"){
          param = c(dataset, data_type, binary, NA, n, p, "l0learn", penalty_type, lamb, g)
        } else {
          param = c(dataset, data_type, binary, fold, n, p, "l0learn", penalty_type, lamb, g)
        }
        out = append(param, results)
        write_log(LogFile, out)
      }
      
    }
  }
}



lasso = function(dataset, data_type, X_train, y_train, X_test, y_test, 
                 LogFile, B, sim_seed, binary, fold){
  print('train lasso')
  n = dim(X_train)[1]
  p = dim(X_train)[2]
  start_time <- Sys.time()
  for (a in c(1, 0.9, 0.7, 0.5, 0.3, 0.1, 0.001)){ 
    model = glmnet(X_train, y_train, family="binomial", alpha=a) # alpha=1 set to lasso
    end_time = Sys.time()
    train_duration = difftime(end_time, start_time, units = "secs")
  
    for (i in 1:length(model$lambda)){
      lamb = model$lambda[i]
      pred_train = predict(model, newx=X_train, s= lamb) # glmnet.predict
      pred_test = predict(model, newx=X_test, s= lamb)
    
      beta = as.vector(coef(model, s=model$lambda[i]))
      penalty_term = lamb * sum(abs(beta[beta != 0])) # l1 loss, glmnet package get l1 penlaty

      results = get_results(beta, pred_train, pred_test, X_train, y_train, X_test, y_test, 
                      data_type, B, penalty_term, train_duration, sim_seed)
      if (data_type=="sim"){
        param = c(dataset, data_type, binary, NA, n, p, "lasso", NA, lamb, a)
      } else {
        param = c(dataset, data_type, binary, fold, n, p, "lasso", NA, lamb, a)
      }
      out = append(param, results)
      write_log(LogFile, out)
    }
  }
}

mcp = function(dataset, data_type, X_train, y_train, X_test, y_test, 
               LogFile, B,sim_seed, binary, fold){
  print('train MCP')
  n = dim(X_train)[1]
  p = dim(X_train)[2]
  for (g in seq(1.5, 25, length=10)){
    start_time <- Sys.time()
    model = ncvreg(X_train, y_train, family = "binomial", penalty="MCP", alpha=1, gamma=g)
    end_time = Sys.time()
    train_duration = difftime(end_time, start_time, units="secs")
    
    for (i in 1:length(model$lambda)){
      lamb = model$lambda[i]
      pred_train = predict(model, X=X_train, lambda = lamb)
      pred_test = predict(model, X=X_test, lambda = lamb)
      
      beta = as.vector(coef(model, lambda=model$lambda[i]))
      results = get_results(beta, pred_train, pred_test, X_train, y_train, X_test, y_test, 
                      data_type, B, penalty_term=0, train_duration=train_duration, sim_seed=sim_seed)
      if (data_type=="sim"){
        param = c(dataset, data_type, binary, NA, n, p, "MCP", NA, lamb, g)
      } else {
        param = c(dataset, data_type, binary, fold, n, p, "MCP", NA, lamb, g)
      }
      out = append(param, results)
      write_log(LogFile, out)
    }
  } 
}


comp_mb_perf = function(dataset, binary, data_type, LogFile, algorithm, ell=NULL, fold=NULL,
                penalty_type=NULL, gammas=NULL,
                sim_seed=NULL, sim_n=NULL, sim_p=NULL, sim_k=NULL){

  data_info = get_dataset(dataset, binary, data_type, fold, 
                      sim_seed, sim_n, sim_p, sim_k)
  X_train = data_info$X_train
  y_train = data_info$y_train
  X_test = data_info$X_test
  y_test = data_info$y_test
  B = data_info$B
  
  if (algorithm == "fastsparse"){
    fs(dataset, data_type, penalty_type, gammas, 
            X_train, y_train, X_test, y_test, LogFile, B, sim_seed, binary, fold, ell)
  } else if (algorithm == "l0learn"){
    l0learn(dataset, data_type, penalty_type, gammas, 
            X_train, y_train, X_test, y_test, LogFile, B, sim_seed, binary, fold)
  } else if (algorithm=='lasso'){
    lasso(dataset, data_type, X_train, y_train, X_test, y_test, LogFile, B, sim_seed, binary, fold)
  } else if (algorithm=='MCP'){
    mcp(dataset, data_type, X_train, y_train, X_test, y_test, LogFile, B, sim_seed, binary, fold)
  } else if (algorithm=='abess'){
    abess_model(dataset, data_type, X_train, y_train, X_test, y_test, LogFile, B, sim_seed, binary, fold)
  }
}



