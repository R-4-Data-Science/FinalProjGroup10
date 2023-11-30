
#data = read.csv("F:/Fall 2023 AU/R Programming for Data Science/Practise/p1/expenses.csv")

data = read.csv("expenses.csv")
library("p1")
# Prepare the predictor matrix X (using 'bmi' as the predictor)
X = matrix(data$bmi, ncol = 1)
X = scale(X)  # Standardizing X

y = ifelse(data$charges > median(data$charges), 1, 0)

if (any(is.na(X), is.infinite(X), is.na(y), is.infinite(y))) {
  stop("Data contains NA or Inf.")
}

# Logistic Regression and subsequent analyses
initial_beta = rep(0, ncol(X))
beta_hat = logistic_regression(X, y)

#bootstrap_conf_intervals(X, y)
plot_logistic_curve(X, beta_hat, y)
confusion_matrix(X, y, beta_hat)
Prevalence(X, y, beta_hat)
Accuracy(X, y, beta_hat)
Sensitivity(X, y, beta_hat)
Specificity(X, y, beta_hat)
False_Discovery_Rate(X, y, beta_hat)
Diagnostic_Odds_ratio(X, y, beta_hat)
prevalencegrid(X, y, beta_hat)
