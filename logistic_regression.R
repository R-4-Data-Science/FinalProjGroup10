# Outline from ChatGPT https://chat.openai.com/share/1a2da067-b3e9-4d8c-932c-c751821bdebb

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

#' Bootstrap Confidence Intervals
#'
#' Function to calculate bootstrap confidence intervals for logistic regression coefficients.
#'
#' @param X Matrix of predictors
#' @param y Binary response variable (0 or 1)
#' @param alpha Significance level for confidence intervals
#' @param n_bootstraps Number of bootstrap samples
#' @return Bootstrap confidence intervals for each coefficient
#' @export
bootstrap_conf_intervals <- function(X, y, alpha = 0.05, n_bootstraps = 20) {
  # Get the number of observations and predictors
  n <- nrow(X)
  p <- ncol(X)
  
  # Initialize an empty matrix to store bootstrap samples
  bootstrap_samples <- matrix(0, nrow = n_bootstrap, ncol = p)
  
  # Perform bootstrap sampling
  set.seed(100)  # Set seed for reproducibility
  for (i in 1:n_bootstrap) {
    # Sample with replacement
    bootstrap_indices <- sample(1:n, replace = TRUE)
    bootstrap_X <- X[bootstrap_indices, ]
    bootstrap_y <- y[bootstrap_indices]
    
    # Fit logistic regression on the bootstrap sample
    fit <- logistic_regression(bootstrap_X, bootstrap_y)
    
    # Store the coefficients in the matrix
    bootstrap_samples[i, ] <- fit
  }
  
  # Compute quantiles for confidence intervals
  lower_quantile <- quantile(bootstrap_samples, alpha / 2)
  upper_quantile <- quantile(bootstrap_samples, 1 - alpha / 2)
  
  # Create a data frame with coefficients and confidence intervals
  ci_data <- data.frame(
    Coefficient = colnames(bootstrap_samples),
    Estimate = colMeans(bootstrap_samples),
    Lower = lower_quantile,
    Upper = upper_quantile
  )
  
  return(ci_data)
}


#' Plot Fitted Logistic Curve
#'
#' Function to plot the fitted logistic curve.
#'
#' @param X Matrix of predictors
#' @param beta Coefficient vector
#' @param y Binary response variable (0 or 1)
#' @export
plot_logistic_curve = function(X, beta, y) {
  # Generate sequence of values for Xbeta
  x_seq <- seq(min(X %*% beta), max(X %*% beta), length.out = 100)
  
  # Calculate predicted probabilities
  pi_seq <- 1 / (1 + exp(-x_seq))
  
  # Plot logistic curve
  plot(X %*% beta, y, pch = 16, col = "black", xlab = "Xbeta", ylab = "Probability")
  lines(x_seq, pi_seq, col = "blue", lwd = 2)
  legend("topright", legend = "Fitted Logistic Curve", col = "blue", lwd = 2)
}

# Add more functions for confusion matrix, evaluation metrics, and plotting metrics over a grid of cutoff values.



set.seed(123)  # Setting seed for reproducibility
n <- 200
p <- 2

X <- matrix(rnorm(n * p), ncol = p)

true_beta <- c(1, -0.5)
pi <- 1 / (1 + exp(-X %*% true_beta))
y <- rbinom(n, 1, pi)

beta_hat <- logistic_regression(X, y)
print(beta_hat)

ci <- bootstrap_conf_intervals(X, y)
print(ci)

plot_logistic_curve(X, beta_hat, y)

