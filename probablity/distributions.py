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

def gamma_function(x):
    """
    Calculate the gamma function for a given value x.

    Parameters:
    x (float): Input value.

    Returns:
    float: Gamma function value.
    """
    return scipy.special.gamma(x)


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


def calculate_multinomial_distribution(x, theta):
    """
    Calculate the multinomial distribution value for given parameters.

    Parameters:
    x (numpy.ndarray): (x_1, ... x_k) numpy array where x_j the number of times side j occurs.
    theta (numpy.ndarray): Array of probabilities for each outcome.

    Returns:
    float: Multinomial distribution value.
    """
    n = np.sum(x)
    coefficient = multinomial_coefficient(n, x)
    theta_factorials = np.prod(theta**x)
    
    result = coefficient * theta_factorials
    return result
    

def bernoulli_distribution(x:int, theta:float) -> float:
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


def multinoulli_distribution(x:np.ndarray, theta:float):
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

    return np.prod([theta[j]**x[j] for j in range(len(x))])



def poisson_distribution(x:np.ndarray, l:float) -> np.ndarray:
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

def gaussian_distribution(X:np.ndarray, *quantiles) ->np.ndarray:
    """
    Calculate the Gaussian (normal) distribution for a set of data points.

    Parameters:
    X (numpy.ndarray): Array of data points.

    Returns:
    numpy.ndarray: Array of Gaussian distribution values.
    """
    if quantiles:
        var = quantiles[1]
        mean = quantiles[0]
    else:
        var = np.var(X)
        mean = np.mean(X)
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(1 / (2 * var)) * (X - mean)**2)


def student_T_distribution(X:np.ndarray, mean:float, sigma_square:float, nu:int):
    """
    Calculate the Student's t-distribution for a set of data points.

    Parameters:
    X (numpy.ndarray): Array of data points.
    *quantiles: Mu(Mean), _sigma_square(Scale parameter)
    nu (int): Degrees of freedom parameter.

    Returns:
    numpy.ndarray: Array of Student's t-distribution values.
    """

    assert(nu > 0 and sigma_square > 0)
    coefficient = gamma_function((nu + 1) / 2) / (np.sqrt(nu*np.pi) * gamma_function(nu/2))
    return coefficient * (1 + ((1/nu) * ((X - mean) / np.sqrt(sigma_square))**2)) **(-((nu + 1)/2))



def laplace_distribution(X:np.ndarray, mean:float, b:float)->np.ndarray:
    """
    Calculate the Laplace distribution for a set of data points.

    Parameters:
    X (numpy.ndarray): Array of data points.
    mu (float): Mean parameter of the Laplace distribution.
    b (float): Scale parameter of the Laplace distribution.

    Returns:
    numpy.ndarray: Array of Laplace distribution values.
    """
    return (1 / (2 * b)) * np.exp(-np.abs(X - mean) / b)

def gamma_distribution(X:np.ndarray, shape:float, rate:float) ->np.ndarray:
    """
    Calculate the Laplace distribution for a set of data points.

    Parameters:
    X (numpy.ndarray): Array of data points positve.
    shape (float > 0):
    rate (float > 0): 

    Returns:
    numpy.ndarray: Array of Gamma distribution values.
    """
    assert(np.all(X >= 0) and shape > 0 and rate > 0)
    return (rate**shape / gamma_function(shape)) * X**(shape-1) * np.exp(-X*rate)

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
    assert(a>0 and b>0)
    B_a_b = (gamma_function(a) * gamma_function(b)) / gamma_function(a + b)
    return X**(a - 1) * (1 - X)**(b - 1) / B_a_b

def pareto_distribution(X:np.ndarray, k:float, m:float) ->np.ndarray:
    """
    Calculate the Pareto distribution functions.
    The pareto distribution is used ti model the distributions of quantities that exhibit long tails

    Parameters:
    X (numpy array): Array of data points
    m (float): constant parameter 
    k (float): control paramter: k -> oo P(x) = dirac(x - m)

    Returns:
    numpy.ndarray: Array of Pareto distribution values
    """

    pareto_dist = np.zeros_like(X)
    mask = X >= m
    pareto_dist[mask] = (k * m**k) /  X[mask]**(k + 1)
    return pareto_dist


def multivariate_gaussian(x, mean, covariance):
    """
    Computes the value of multivariate Gaussian of a distribution at point x

    Parameters:
    -x: Point at which to compute the probablity density function.
    -mean: mean vector of the distribution
    -covariance: Covariance matrix of the distribution.

    Returns: 
    -Value of the multivariate Gaussion distribution at x
    """
    #assert(len(mean == x.shape))
    d = len(mean)
    coeff = 1 / np.sqrt((2 * np.pi)**d * np.linalg.det(covariance))
    exponent = -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(covariance)), (x - mean))
    return coeff * np.exp(exponent)


def multivariate_StudentT_distribution(x, mean, scale_matrix, nu):
    """
    Robust alternative to the MVN

    Parameters:
    -x: Point at which to compute the probablity desnity function.
    -mean: mean vector of the distribution
    -scale_matrix: Scale Matrix
    nu:
    """

    d = len(mean)
    coeff1 = (gamma_function(nu/2 + d/2) / gamma_function(nu/2)) * (1/np.sqrt(np.pi*np.linalg.det(nu*scale_matrix)))
    coeff2 = (1 + np.dot(np.dot((x - mean).T, np.linalg.inv(nu*scale_matrix)), (x - mean)))** (-(nu + d) / 2)

    return coeff1 * coeff2