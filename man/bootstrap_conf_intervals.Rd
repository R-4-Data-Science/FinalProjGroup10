% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/Con_in.R
\name{bootstrap_conf_intervals}
\alias{bootstrap_conf_intervals}
\title{This function calculates bootstrap confidence intervals for logistic regression coefficients.}
\usage{
bootstrap_conf_intervals(X, y, alpha = 0.05, n_bootstraps = 20)
}
\arguments{
\item{X}{A matrix of predictors.}

\item{y}{A binary response variable.}

\item{alpha}{Significance level for the confidence intervals.}

\item{n_bootstraps}{Number of bootstrap samples.}
}
\value{
A matrix of bootstrap confidence intervals for each coefficient.
}
\description{
This function calculates bootstrap confidence intervals for logistic regression coefficients.
}
