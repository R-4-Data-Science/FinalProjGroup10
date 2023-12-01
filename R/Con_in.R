#' Logistic Regression
#'
#' Function to perform logistic regression using numerical optimization.
#'
#' @param X Matrix of predictors
#' @param y Binary response variable (0 or 1)
#' @return Coefficient vector beta
#' @export
logistic_regression = function(X, y) {

  initial_beta = solve(t(X)%*%X)%*%t(X)%*%y

  result = optim(par = initial_beta, fn = logistic_loss, X = X, y = y, method = "BFGS")

  beta_hat = as.numeric(result$par[, 1])

  return(beta_hat)
}

# Helper function to calculate logistic loss
logistic_loss = function(beta, X, y) {
  pi = 1 / (1 + exp(-X %*% beta))
  loss = -sum(y * log(pi) + (1 - y) * log(1 - pi))
  return(loss)
}


#' This function calculates bootstrap confidence intervals for logistic regression coefficients.
#'
#' @param X A matrix of predictors.
#' @param y A binary response variable.
#' @param alpha Significance level for the confidence intervals.
#' @param n_bootstraps Number of bootstrap samples.
#' @return A matrix of bootstrap confidence intervals for each coefficient.
#' @export
bootstrap_conf_intervals = function(X, y, alpha = 0.05, n_bootstraps = 20) {
  n = nrow(X)
  p = ncol(X)

  # Initialize an empty matrix to store bootstrap samples
  b_samples = matrix(0, p, n_bootstraps)

  # Perform bootstrap sampling
  set.seed(100)  # Set seed for reproducibility
  for (b in 1:n_bootstraps) {
    # Sample with replacement
    b_indices = sample(1:n, n, replace = TRUE)
    b_X = X[b_indices,]
    b_y = y[b_indices]

    b_samples[, b] = logistic_regression(b_X, b_y)
  }

  # Compute quantiles for confidence intervals
  lower = apply(b_samples, 1, quantile, 1-alpha)
  upper = apply(b_samples, 1, quantile, alpha)


  ci = cbind(lower, upper)
  return(ci)
}

#' Plot Fitted Logistic Curve
#'
#' Function to plot the fitted logistic curve.
#'
#' @param X Matrix of predictors
#' @param beta_hat Coefficient vector
#' @param y Binary response variable (0 or 1)
#' @export
plot_logistic_curve = function(X, beta_hat, y) {
  # Generate sequence of values for Xbeta
  x_seq = seq(min(X %*% beta_hat), max(X %*% beta_hat), length.out = 100)

  # Calculate predicted probabilities
  pi_seq = 1 / (1 + exp(-x_seq))

  # Plot logistic curve
  plot(X %*% beta_hat, y, pch = 16, col = "black", xlab = "Xbeta", ylab = "Probability")
  lines(x_seq, pi_seq, col = "blue", lwd = 2)
}

#' Confusion Matrix
#'
#' Confusion matrix with a cut-off value for predictions at 0.5. Predictions above 0.5 are assigned a 1, below 0.5 are assigned 0.
#'
#' @param X Matrix of predictors
#' @param beta_hat Coefficient vector
#' @param y Binary response variable (0 or 1)
#' @param cut Cut off value
#' @return Confidence Matrix
#' @export
confusion_matrix = function(X, y, beta_hat, cut = 0.5){

  yhat = (1/(1+exp(-X%*%beta_hat)) > 0.5)*1

  tp = sum((yhat==1)&(y==1))
  fp = sum((yhat==0)&(y==1))
  tn = sum((yhat==0)&(y==0))
  fn = sum((yhat==1)&(y==0))
  conf_mat = cbind(c(tp, fp), c(fn, tn))

  return(conf_mat)
}

#' Prevalence
#'
#' Calculates the prevalence based on the confusion matrix.
#'
#' @param X Matrix of predictors
#' @param y Binary response variable (0 or 1)
#' @param beta_hat Coefficient vector
#' @param cut Cut off value
#' @return Prevalence
#' @export
Prevalence = function(X, y, beta_hat, cut = 0.5){

  yhat = (1/(1+exp(-X%*%beta_hat)) > 0.5)*1

  tp = sum((yhat==1)&(y==1))
  fp = sum((yhat==0)&(y==1))
  tn = sum((yhat==0)&(y==0))
  fn = sum((yhat==1)&(y==0))

  Prevalence = (fn+tp)/(tp+fp+tn+fn)

  return(Prevalence)
}


#' Accuracy
#'
#' Calculates the accuracy based on the confusion matrix.
#'
#' @param X Matrix of predictors
#' @param y Binary response variable (0 or 1)
#' @param beta_hat Coefficient vector
#' @param cut Cut off value
#' @return Accuracy
#' @export
Accuracy = function(X, y, beta_hat, cut = 0.5){

  yhat = (1/(1+exp(-X%*%beta_hat)) > 0.5)*1
  tp = sum((yhat==1)&(y==1))
  fp = sum((yhat==0)&(y==1))
  tn = sum((yhat==0)&(y==0))
  fn = sum((yhat==1)&(y==0))

  Accuracy = (tn+tp)/(tp+fp+tn+fn)

  return(Accuracy)
}

#' Sensitivity
#'
#' Calculates the sensitivity based on the confusion matrix.
#'
#' @param X Matrix of predictors
#' @param y Binary response variable (0 or 1)
#' @param beta_hat Coefficient vector
#' @param cut Cut off value
#' @return Sensitivity
#' @export
Sensitivity = function(X, y, beta_hat, cut = 0.5){

  yhat = (1/(1+exp(-X%*%beta_hat)) > 0.5)*1
  tp = sum((yhat==1)&(y==1))
  fp = sum((yhat==0)&(y==1))
  tn = sum((yhat==0)&(y==0))
  fn = sum((yhat==1)&(y==0))

  Sensitivity = tp/(tp+fn)

  return(Sensitivity)
}

#' Specificity
#'
#' Calculates the specificity based on the confusion matrix.
#'
#' @param X Matrix of predictors
#' @param y Binary response variable (0 or 1)
#' @param beta_hat Coefficient vector
#' @param cut Cut off value
#' @return Specificity
#' @export
Specificity = function(X, y, beta_hat, cut = 0.5){

  yhat = (1/(1+exp(-X%*%beta_hat)) > 0.5)*1
  tp = sum((yhat==1)&(y==1))
  fp = sum((yhat==0)&(y==1))
  tn = sum((yhat==0)&(y==0))
  fn = sum((yhat==1)&(y==0))

  Specificity = tn/(tn+fp)

  return(Specificity)
}

#' False Discovery Rate
#'
#' Calculates the false discovery rate based on the confusion matrix.
#'
#' @param X Matrix of predictors
#' @param y Binary response variable (0 or 1)
#' @param beta_hat Coefficient vector
#' @param cut Cut off value
#' @return False Discovery Rate
#' @export
False_Discovery_Rate = function(X, y, beta_hat, cut = 0.5){

  yhat = (1/(1+exp(-X%*%beta_hat)) > 0.5)*1
  tp = sum((yhat==1)&(y==1))
  fp = sum((yhat==0)&(y==1))
  tn = sum((yhat==0)&(y==0))
  fn = sum((yhat==1)&(y==0))

  False_Discovery_Rate = fp/(tp+fp)

  return(False_Discovery_Rate)
}

#' Diagnostic Odds Ratio
#'
#' Calculates the diagnostic odds ratio based on the confusion matrix.
#'
#' @param X Matrix of predictors
#' @param y Binary response variable (0 or 1)
#' @param beta_hat Coefficient vector
#' @param cut Cut off value
#' @return Diagnostic Odds Ratio
#' @export
Diagnostic_Odds_ratio = function(X, y, beta_hat, cut = 0.5){

  yhat = (1/(1+exp(-X%*%beta_hat)) > 0.5)*1
  tp = sum((yhat==1)&(y==1))
  fp = sum((yhat==0)&(y==1))
  tn = sum((yhat==0)&(y==0))
  fn = sum((yhat==1)&(y==0))

  Diagnostic_Odds_ratio = (Sensitivity(X,y,beta_hat) / False_Discovery_Rate(X,y,beta_hat)) / (False_Discovery_Rate(X,y,beta_hat)/ Specificity(X,y,beta_hat))


  return(Diagnostic_Odds_ratio)
}

#' Prevalence Grid
#'
#' Calculates prevalence for different cut-off values and returns a matrix.
#'
#' @param X Matrix of predictors
#' @param y Binary response variable (0 or 1)
#' @param beta_hat Coefficient vector
#' @param cut Vector of cut-off values
#' @return Matrix of prevalence values for each cut-off
#' @export
prevalencegrid <- function(X, y, beta_hat, cut = seq(0.1, 0.9, by = 0.1)) {
  n_cuts = length(cut)
  metrics_matrix = matrix(NA, n_cuts, 7,
                          dimnames = list(NULL, c("Cut-Off", "Prevalence", "Accuracy", "Sensitivity", "Specificity", "False_Discovery_Rate", "Diagnostic_Odds_Ratio")))

  for (i in 1:n_cuts) {
    current_cut = cut[i]
    yhat = (1 / (1 + exp(-X %*% beta_hat)) > current_cut) * 1

    metrics_matrix[i, 1] = current_cut
    metrics_matrix[i, 2] = Prevalence(X, y, beta_hat, current_cut)
    metrics_matrix[i, 3] = Accuracy(X, y, beta_hat, current_cut)
    metrics_matrix[i, 4] = Sensitivity(X, y, beta_hat, current_cut)
    metrics_matrix[i, 5] = Specificity(X, y, beta_hat, current_cut)
    metrics_matrix[i, 6] = False_Discovery_Rate(X, y, beta_hat, current_cut)
    metrics_matrix[i, 7] = Diagnostic_Odds_ratio(X, y, beta_hat, current_cut)
  }




  return(metrics_matrix)
}
