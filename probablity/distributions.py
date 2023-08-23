"""
CogniCraft Probability Distributions Module

This module provides functions for calculating probabilities using various probability distributions.
It includes functions for binomial and Bernoulli distributions.

Functions:
- binomial_coefficient(n, k): Calculate the binomial coefficient (n choose k).
- multinomial_coefficient(n, x): Calculate the number of ways to divide a set of size n = sum(k=1 -> K)x_k into subsets with sizes x1 up to xK.
- binomial_distribution(n, k, theta): Calculate the probability of k successes in n trials using the binomial distribution.
- calculate_multinomial_distribution(n, x, theta): Calculate the multinomial distribution value for given parameters.
- bernoulli_distribution(x, theta): Calculate the probability of a Bernoulli trial.
- multinoulli_distribution(x, theta): Calculate the multinoulli distribution value for a given outcome vector and probability vector.
- poisson_distribution(x, l): Calculate the probability of observing x events in a Poisson distribution.
- empirical_distribution(D, A): Calculate the empirical distribution value for a set of data and a subset A.
- gaussian_distribution(X): Calculate the Gaussian (normal) distribution for a set of data points.
- student_T_distribution(X, df): Calculate the Student's t-distribution for a set of data points.
- laplace_distribution(X, mu, b): Calculate the Laplace distribution for a set of data points.
- beta_distribution(X, a, b): Calculate the Beta distribution for a set of data points.
- multivariate_normal(X, mean, cov_matrix): Calculate the multivariate normal of a set of data points
- gamma_function(x): Calculate the gamma function for a given value x.
Author: Casteck Axma
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.special

def binomial_coefficient(n, k):
    """
    Calculate the binomial coefficient (n choose k).

    Parameters:
    n (int): Total number of trials.
    k (int): Number of successful outcomes.

    Returns:
    float: Binomial coefficient value.
    """
    return np.math.factorial(n) / (np.math.factorial(k) * np.math.factorial(n - k))


def multinomial_coefficient(n, x):
    """
    Calculate the multinomial coefficient (n choose x1, x2, ... xn).

    Parameters:
    n (int): Total number of trials.
    x (numpy.ndarray): Array of counts for each outcome.

    Returns:
    float: Multinomial coefficient value.
    """
    x_factorials = np.prod([np.math.factorial(xi) for xi in x])
    return np.math.factorial(n) / x_factorials

def binomial_distribution(n, k, theta):
    """
    Calculate the probability of k successes in n trials using the binomial distribution.

    Parameters:
    n (int): Total number of trials.
    k (int): Number of successful outcomes.
    theta (float): Probability of success in a single trial.

    Returns:
    float: Probability of k successes.
    """
    return binomial_coefficient(n, k) * (theta ** k) * ((1 - theta) ** (n - k))


def calculate_multinomial_distribution(n, x, theta):
    """
    Calculate the multinomial distribution value for given parameters.

    Parameters:
    n (int): Total number of trials.
    x (numpy.ndarray): Array of counts for each outcome.
    theta (numpy.ndarray): Array of probabilities for each outcome.

    Returns:
    float: Multinomial distribution value.
    """
    coefficient = multinomial_coefficient(n, x)
    theta_factorials = np.prod(theta**x)
    
    result = coefficient * theta_factorials
    return result
    

def bernoulli_distribution(x, theta):
    """
    Calculate the probability of a Bernoulli trial.

    Parameters:
    x (int): Outcome of the Bernoulli trial (0 or 1).
    theta (float): Probability of success (1) in a single trial.

    Returns:
    float: Probability of the given outcome.
    """
    if x == 1:
        return theta
    return 1 - theta


def multinoulli_distribution(x, theta):
    """
    Calculate the multinoulli distribution value for a given outcome vector and probability vector.

    Parameters:
    x (numpy.ndarray): Outcome vector (0 or 1 for each category).
    theta (numpy.ndarray): Probability vector for each category.

    Returns:
    float: Multinoulli distribution value.
    """
    # Make sure x and theta have the same length
    if len(x) != len(theta):
        raise ValueError("x and theta must have the same length")

    # Calculate the product of theta[j] for each j where x[j] == 1
    return np.prod([theta[j] for j in range(len(x)) if x[j] == 1])



def poisson_distribution(x, l):
    """
    Calculate the probability of observing x events in a Poisson distribution.

    Parameters:
    x (int): Number of events.
    l (float): Average rate of events.

    Returns:
    float: Poisson distribution value.
    """
    return np.exp(-l) * (l ** x / np.math.factorial(x))

def empirical_distribution(D, A):
    """
    Calculate the empirical distribution value for a set of data and a subset A.

    Parameters:
    D (numpy.ndarray): Array of data points.
    A (set): Subset of data points.

    Returns:
    float: Empirical distribution value.
    """
    def dirac_measure(x, A):
        if x in A:
            return 1
        return 0

    return (np.sum([dirac_measure(x, A) for x in D])) / len(D)

def gaussian_distribution(X):
    """
    Calculate the Gaussian (normal) distribution for a set of data points.

    Parameters:
    X (numpy.ndarray): Array of data points.

    Returns:
    numpy.ndarray: Array of Gaussian distribution values.
    """
    var = np.var(X)
    mean = np.mean(X)
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(1 / (2 * var)) * (X - mean)**2)


def student_T_distribution(X, df):
    """
    Calculate the Student's t-distribution for a set of data points.

    Parameters:
    X (numpy.ndarray): Array of data points.
    df (int): Degrees of freedom parameter.

    Returns:
    numpy.ndarray: Array of Student's t-distribution values.
    """
    constant = gamma_function(df + 1 / 2) / (np.sqrt(df * np.pi) * gamma_function(df / 2))
    pdf_values = constant * (1 + (X**2 / df))**(-0.5 * (df + 1))
    return pdf_values



def laplace_distribution(X, mu, b):
    """
    Calculate the Laplace distribution for a set of data points.

    Parameters:
    X (numpy.ndarray): Array of data points.
    mu (float): Mean parameter of the Laplace distribution.
    b (float): Scale parameter of the Laplace distribution.

    Returns:
    numpy.ndarray: Array of Laplace distribution values.
    """
    return (1 / (2 * b)) * np.exp(-np.abs(X - mu) / b)


def beta_distribution(X, a, b):
    """
    Calculate the Beta distribution for a set of data points.

    Parameters:
    X (numpy.ndarray): Array of data points.
    a (float): Shape parameter a (alpha) for the Beta distribution.
    b (float): Shape parameter b (beta) for the Beta distribution.

    Returns:
    numpy.ndarray: Array of Beta distribution values.
    """
    B_a_b = (gamma_function(a) * gamma_function(b)) / gamma_function(a + b)
    return X**(a - 1) * (1 - X)**(b - 1) / B_a_b
import numpy as np

def multivariate_normal(X, mean, cov_matrix):
    """
    Calculate the multivariate normal probability density function.

    This function computes the probability density function of a multivariate normal distribution
    given the data points, mean, and covariance matrix.

    Parameters:
    X (numpy.ndarray): Data points as a matrix of shape (n_samples, n_features).
    mean (numpy.ndarray): Mean vector of the distribution with shape (n_features,).
    cov_matrix (numpy.ndarray): Covariance matrix of the distribution with shape (n_features, n_features).

    Returns:
    numpy.ndarray: Array of probability density values for each data point.

    Example:
    X = np.array([[1, 2], [3, 4], [5, 6]])
    mean = np.array([2, 3])
    cov_matrix = np.array([[1, 0.5], [0.5, 1]])
    probabilities = multivariate_normal(X, mean, cov_matrix)
    """
    if mean.shape[0] != cov_matrix.shape[0]:
        raise ValueError("Mean vector dimension must match the dimension of the covariance matrix")

    k = mean.shape[0]
    det_cov = np.linalg.det(cov_matrix)
    if det_cov <= 0:
        raise ValueError("Covariance matrix must be positive definite")

    constant = 1.0 / ((2 * np.pi) ** (0.5 * k) * det_cov)

    X_minus_mean = X - mean
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    exponent = -0.5 * np.sum(X_minus_mean.dot(inv_cov_matrix) * X_minus_mean, axis=1)
    probabilities = constant * np.exp(exponent)
    return probabilities

def gamma_function(x):
    """
    Calculate the gamma function for a given value x.

    Parameters:
    x (float): Input value.

    Returns:
    float: Gamma function value.
    """
    return scipy.special.gamma(x)

