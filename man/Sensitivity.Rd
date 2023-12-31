% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/Con_in.R
\name{Sensitivity}
\alias{Sensitivity}
\title{Sensitivity}
\usage{
Sensitivity(X, y, beta_hat, cut = 0.5)
}
\arguments{
\item{X}{Matrix of predictors}

\item{y}{Binary response variable (0 or 1)}

\item{beta_hat}{Coefficient vector}

\item{cut}{Cut off value}
}
\value{
Sensitivity
}
\description{
Calculates the sensitivity based on the confusion matrix.
}
