library(dplyr)
library(pROC)
library(reticulate)
library(MASS)

get_auc = function(y_true, y_pred){
  y_true = factor(y_true)
  y_pred = as.vector(y_pred)
  if (sum(is.nan(y_pred)) > 0) {
    return(NA)
  } else{
    roccurve = roc(y_true ~ y_pred)
    return(as.numeric(roccurve$auc))
  }
}

get_acc = function(y_true, y_pred){
  y_pred = as.vector(y_pred)
  if (sum(is.nan(y_pred))>0){
    return(NA)
  } else {
    y_pred[y_pred>=0.5]=1
    y_pred[y_pred<0.5]= -1
    loss = sum(y_true != y_pred)/length(y_true)
    return(1-loss)
  }
}

get_logistic_loss = function(intercept, b, X, y){
  if (sum(is.nan(b)) > 0){
    return(NA)
  } else {
    wx = X %*% b + intercept
    loss = sum(log(exp(wx*y * -1) + 1))
    return(loss)
  }
}

get_recover_f1 = function(beta_true, beta_fit){
  # beta_true and beta_fit should be vectors 
  if (sum(is.nan(beta_fit)) > 0) {
    returnlist = listlist("precision" = NA, "recall"=NA, "f1"=NA)
    return(returnlist)
  } else {
    count_intersect = sum((beta_true != 0) & (beta_fit != 0))
    p = count_intersect/sum(beta_fit != 0)
    r = count_intersect/sum(beta_true != 0)
    returnlist = list("precision" = p, "recall"=r, "f1"=2*p*r/(p+r))
    return(returnlist)
  }
}



get_dataset = function(train_path, test_path){
  # dataset is a string
  np = import("numpy") 
  mat_train <- np$load(train_path)
  mat_test = np$load(test_path)
  
  p = dim(mat_train)[2]
  
  X_train = mat_train[,1:p-1]
  y_train = mat_train[,p]
  
  X_test = mat_test[,1:p-1]
  y_test = mat_test[,p]
  
  returnlist = list("X_train" = X_train, "y_train" = y_train, 
                    "X_test" = X_test, "y_test" = y_test)
  
  return(returnlist)
}


get_dataset = function(dataset, binary, data_type, fold=NULL, 
                      sim_seed=NULL, sim_n=NULL, sim_p=NULL, sim_k=NULL){
  # dataset is a string
  np = import("numpy") 
  if (data_type=="real"){
    if (binary){
      train_path = paste("datasets/",dataset,"/", dataset, "_bin_train", fold,".npy", sep="")
      test_path = paste("datasets/",dataset,"/", dataset, "_bin_test", fold,".npy", sep="")
    } else{
      train_path = paste("datasets/", dataset, "/", dataset, "_train", fold,".npy", sep="")
      test_path = paste("datasets/", dataset, "/", dataset, "_test", fold,".npy", sep="")
    }
    B = NULL
  } else{
    train_path = paste(dataset, "train", sim_n, sim_p, sim_k, sim_seed, sep="_")
    train_path = paste("datasets/", dataset, "/", train_path, ".npy", sep="")
    test_path = paste(dataset, "test", sim_n, sim_p, sim_k, sim_seed, sep="_")
    test_path = paste("datasets/", dataset, "/", test_path, ".npy", sep="")
    B = c(rep(0, sim_p))
    step = floor(sim_p/sim_k)
    for (l in 1:sim_k){
      B[l*step] = 1
    }
  }
  

  mat_train <- np$load(train_path)
  mat_test = np$load(test_path)
  
  p = dim(mat_train)[2]
  
  X_train = mat_train[,1:p-1]
  y_train = mat_train[,p]
  
  X_test = mat_test[,1:p-1]
  y_test = mat_test[,p]
  
  returnlist = list("X_train" = X_train, "y_train" = y_train, 
                    "X_test" = X_test, "y_test" = y_test, "B" = B)
  
  return(returnlist)
}


get_norm_from_centeredX = function(X){ # l0study X transform 
  Xmean = colMeans(X)
  Xcentered = sweep(X, 2, Xmean)
  Xcentered_squared = Xcentered * Xcentered
  Xnorm = sqrt(colSums(Xcentered_squared))
  return(Xnorm)
}


write_log = function(LogFile, out){
  if (!file.exists(LogFile)){
    colnames = c("dataset", "data_type", "binary_feature", "fold", "n", "p", 
                "algorithm", "penalty_type", "lamb", "g", "simulation_seed", "support_size", 
                "train_auc", "test_auc", 
                "train_acc", "test_acc", "train_log_loss", "test_log_loss", "train_obj", "test_obj", 
                "train_precision", "train_recall", "train_f1_score", 
                "train_exp_loss", "test_exp_loss","train_duration", "\n")
    cat(colnames, file=LogFile, append=TRUE, sep=";")
  }
  cat(out, file=LogFile, append=TRUE, sep=";")
}


get_results = function(beta, pred_train, pred_test, X_train, y_train, X_test, y_test, 
                      data_type, B, penalty_term, train_duration, sim_seed){
  intercept = beta[1]
  b = beta[2:length(beta)]  
  support = sum(b!=0)
  train_log_loss = get_logistic_loss(intercept, b, X_train, y_train)
  test_log_loss = get_logistic_loss(intercept, b, X_test, y_test)
  train_obj = train_log_loss + penalty_term
  test_obj = test_log_loss + penalty_term
  exp_loss_train = sum(exp(-y_train * (X_train %*% b + intercept)))
  exp_loss_test = sum(exp(-y_test * (X_test %*% b + intercept)))

  results = c(support, get_auc(y_train, pred_train), get_auc(y_test, pred_test), 
              get_acc(y_train, pred_train), get_acc(y_test, pred_test), 
              train_log_loss, test_log_loss, train_obj, test_obj)

  if (data_type == "real"){
    seed = NA
    more_results = c(NA, NA, NA, exp_loss_train, exp_loss_test, train_duration, "\n")
  } else{
    seed = sim_seed
    f1return = get_recover_f1(B, b)
    more_results = c(f1return$precision, f1return$recall, f1return$f1, exp_loss_train, exp_loss_test, train_duration, "\n")
  }
  
  out = append(seed, results)
  out = append(out, more_results)
  
  return(out)
}
