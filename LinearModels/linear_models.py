import numpy as np
from tqdm import tqdm

class LinearRegression:

    """
    A simple implementation of linear regression using NumPy.

    This class provides methods to fit a linear regression model to input data,
    predict target values, and calculate mean squared error.

    Attributes:
        best_theta (numpy.ndarray): Best-fit parameters of the linear regression model.

    Methods:
        normalize(X): Normalize the input features using z-score normalization.
        fit(X, y, normalize=False): Fit a linear regression model to the input data.
        fit_intercept(X, y): Fit a linear regression model with an intercept term.
        predict(X): Make predictions using the fitted linear regression model.
        mean_squared_error(y_true, y_pred): Calculate mean squared error between true and predicted values.
    """

    def __init__(self):
        """
        Initialize the LinearRegression class.
        """
        self.best_theta = []

    def normalize(self, X):
        """
        Normalize the input features using z-score normalization.

        Parameters:
        X (numpy.ndarray): Input feature matrix.

        Returns:
        numpy.ndarray: Normalized feature matrix.
        """
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        return (X - X_mean) / X_std

    def fit(self, X, y, normalize=False):
        """
        Fit a linear regression model to the input data.

        Parameters:
        X (numpy.ndarray): Input feature matrix.
        y (numpy.ndarray): Target values.
        normalize (bool): Whether to normalize the features.

        Returns:
        None
        """
        if normalize:
            X = self.normalize(X)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.best_theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    def fit_intercept(self, X, y):
        """
        Fit a linear regression model with an intercept term.

        Parameters:
        X (numpy.ndarray): Input feature matrix.
        y (numpy.ndarray): Target values.

        Returns:
        numpy.ndarray: Best-fit parameters (including intercept).
        """
        self.fit(X, y)
        return self.best_theta

    def predict(self, X):
        """
        Make predictions using the fitted linear regression model.

        Parameters:
        X (numpy.ndarray): Input feature matrix for prediction.

        Returns:
        numpy.ndarray: Predicted target values.
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.best_theta)

    def mean_squared_error(self, y_true, y_pred):
        """
        Calculate the mean squared error between true and predicted values.

        Parameters:
        y_true (numpy.ndarray): True target values.
        y_pred (numpy.ndarray): Predicted target values.

        Returns:
        float: Mean squared error.
        """
        return np.mean((y_true - y_pred) ** 2)
    

class PolynomialRegression:

    """
    A simple implementation of polynomial regression using NumPy.

    This class provides methods to fit a polynomial regression model to input data,
    predict target values, and calculate coefficients for polynomial terms.

    Attributes:
        degree (int): Degree of the polynomial regression.
        coef (numpy.ndarray): Coefficients of the polynomial regression model.

    Methods:
        polynomial_features(X): Transform input features into polynomial features.
        fit(X, y): Fit a polynomial regression model to the input data.
        predict(X): Make predictions using the fitted polynomial regression model.
    """

    def __init__(self, degree=1):
        """
        Initialize the PolynomialRegression class.

        Parameters:
            degree (int): Degree of the polynomial regression (default is 1).
        """
        self.degree = degree
        self.coef = []

    def polynomial_features(self, X):
        """
        Transform input features into polynomial features.

        Parameters:
            X (numpy.ndarray): Input feature matrix.

        Returns:
            numpy.ndarray: Transformed feature matrix with polynomial terms.
        """
        return np.column_stack([X**i for i in range(1, self.degree + 1)])

    def fit(self, X, y):
        """
        Fit a polynomial regression model to the input data.

        Parameters:
            X (numpy.ndarray): Input feature matrix.
            y (numpy.ndarray): Target values.

        Returns:
            None
        """
        X_poly = self.polynomial_features(X)
        X_b = np.c_[np.ones((X_poly.shape[0], 1)), X_poly]
        self.coef = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    def predict(self, X):
        """
        Make predictions using the fitted polynomial regression model.

        Parameters:
            X (numpy.ndarray): Input feature matrix for prediction.

        Returns:
            numpy.ndarray: Predicted target values.
        """
        X_poly = self.polynomial_features(X)
        X_b = np.c_[np.ones((X_poly.shape[0], 1)), X_poly]
        return X_b.dot(self.coef)
    


class RidgeRegression:
    """
    A simple implementation of ridge regression using NumPy.

    This class provides methods to fit a ridge regression model to input data,
    predict target values, and calculate coefficients with regularization.

    Attributes:
        alpha (float): Regularization parameter.
        coef (numpy.ndarray): Coefficients of the ridge regression model.

    Methods:
        fit(X, y): Fit a ridge regression model to the input data.
        predict(X): Make predictions using the fitted ridge regression model.
    """

    def __init__(self, alpha=1):
        """
        Initialize the RidgeRegression class.

        Parameters:
            alpha (float): Regularization parameter (default is 1).
        """
        self.alpha = alpha
        self.coef = []

    def fit(self, X, y):
        """
        Fit a ridge regression model to the input data.

        Parameters:
            X (numpy.ndarray): Input feature matrix.
            y (numpy.ndarray): Target values.

        Returns:
            None
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        I = np.identity(X_b.shape[1])
        I[0][0] = 0  # Exclude regularization for bias term
        self.coef = np.linalg.inv(X_b.T.dot(X_b) + self.alpha * I).dot(X_b.T).dot(y)

    def predict(self, X):
        """
        Make predictions using the fitted ridge regression model.

        Parameters:
            X (numpy.ndarray): Input feature matrix for prediction.

        Returns:
            numpy.ndarray: Predicted target values.
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.coef)

class PolynomialRidgeRegression:
    """
    A simple implementation of polynomial ridge regression using NumPy.

    This class provides methods to fit a polynomial ridge regression model to input data,
    predict target values, calculate coefficients with regularization, and compute mean squared error.

    Attributes:
        degree (int): Degree of the polynomial.
        alpha (float): Regularization parameter.
        coef (numpy.ndarray): Coefficients of the polynomial ridge regression model.

    Methods:
        polynomial_features(X): Transform input features into polynomial features.
        fit(X, y): Fit a polynomial ridge regression model to the input data.
        predict(X): Make predictions using the fitted polynomial ridge regression model.
        mean_squared_error(y_true, y_pred): Calculate mean squared error between true and predicted values.
    """

    def __init__(self, degree=1, alpha=1):
        """
        Initialize the PolynomialRidgeRegression class.

        Parameters:
            degree (int): Degree of the polynomial (default is 1).
            alpha (float): Regularization parameter (default is 1).
        """
        self.degree = degree
        self.alpha = alpha
        self.coef = []

    def polynomial_features(self, X):
        """
        Transform input features into polynomial features.

        Parameters:
            X (numpy.ndarray): Input feature matrix.

        Returns:
            numpy.ndarray: Transformed feature matrix with polynomial terms.
        """
        return np.column_stack([X**i for i in range(1, self.degree + 1)])

    def fit(self, X, y):
        """
        Fit a polynomial ridge regression model to the input data.

        Parameters:
            X (numpy.ndarray): Input feature matrix.
            y (numpy.ndarray): Target values.

        Returns:
            None
        """
        X_poly = self.polynomial_features(X)
        X_b = np.c_[np.ones((X_poly.shape[0], 1)), X_poly]
        I = np.identity(X_b.shape[1])
        I[0][0] = 0  # Exclude regularization for bias term
        self.coef = np.linalg.inv(X_b.T.dot(X_b) + self.alpha * I).dot(X_b.T).dot(y)

    def predict(self, X):
        """
        Make predictions using the fitted polynomial ridge regression model.

        Parameters:
            X (numpy.ndarray): Input feature matrix for prediction.

        Returns:
            numpy.ndarray: Predicted target values.
        """
        X_poly = self.polynomial_features(X)
        X_b = np.c_[np.ones((X_poly.shape[0], 1)), X_poly]
        return X_b.dot(self.coef)
    


class LassoRegression:
    """
    A simple implementation of Lasso linear regression using NumPy.

    This class provides methods to fit a Lasso linear regression model to input data,
    predict target values, calculate coefficients with L1 regularization,
    and compute mean squared error.

    Attributes:
        alpha (float): Regularization parameter for L1 penalty.
        coef (numpy.ndarray): Coefficients of the Lasso linear regression model.

    Methods:
        fit(X, y): Fit a Lasso linear regression model to the input data.
        predict(X): Make predictions using the fitted Lasso linear regression model.
        mean_squared_error(y_true, y_pred): Calculate mean squared error between true and predicted values.
    """

    def __init__(self, alpha=1.0):
        """
        Initialize the LassoRegression class.

        Parameters:
            alpha (float): Regularization parameter (default is 1.0).
        """
        self.alpha = alpha
        self.coef = []

    def soft_threshold(self, x, threshold):
        """
        Perform soft thresholding.

        Parameters:
            x (float): Input value.
            threshold (float): Threshold value.

        Returns:
            float: Soft thresholded value.
        """
        if x > threshold:
            return x - threshold
        elif x < -threshold:
            return x + threshold
        else:
            return 0

    def fit(self, X, y, max_iterations=100, tol=1e-4):
        """
        Fit a Lasso linear regression model to the input data.

        Parameters:
            X (numpy.ndarray): Input feature matrix.
            y (numpy.ndarray): Target values.
            max_iterations (int): Maximum number of iterations for coordinate descent.
            tol (float): Tolerance for convergence.

        Returns:
            None
        """
        n_samples, n_features = X.shape
        self.coef = np.zeros(n_features)
        XTX = X.T.dot(X)

        for _ in range(max_iterations):
            old_coef = np.copy(self.coef)
            for j in range(n_features):
                X_j = X[:, j]
                y_pred = X.dot(self.coef)
                r_j = X_j.T.dot(y - y_pred + self.coef[j] * X_j)
                l1_update = self.soft_threshold(r_j, self.alpha)
                self.coef[j] = l1_update / XTX[j, j]

            if np.linalg.norm(self.coef - old_coef) < tol:
                break

    def predict(self, X):
        """
        Make predictions using the fitted Lasso linear regression model.

        Parameters:
            X (numpy.ndarray): Input feature matrix for prediction.

        Returns:
            numpy.ndarray: Predicted target values.
        """
        return X.dot(self.coef)

import numpy as np

class ElasticNetRegression:
    """
    A simple implementation of Elastic Net regression using NumPy.

    This class provides methods to fit an Elastic Net regression model to input data,
    predict target values, calculate coefficients with both L1 and L2 regularization,
    and compute mean squared error.

    Attributes:
        alpha (float): Regularization parameter for the combined L1 and L2 penalties.
        l1_ratio (float): Mixing parameter between L1 and L2 penalties.
        coef (numpy.ndarray): Coefficients of the Elastic Net regression model.

    Methods:
        fit(X, y): Fit an Elastic Net regression model to the input data.
        predict(X): Make predictions using the fitted Elastic Net regression model.
        mean_squared_error(y_true, y_pred): Calculate mean squared error between true and predicted values.
    """

    def __init__(self, alpha=1.0, l1_ratio=0.5):
        """
        Initialize the ElasticNetRegression class.

        Parameters:
            alpha (float): Regularization parameter (default is 1.0).
            l1_ratio (float): Mixing parameter between L1 and L2 penalties (default is 0.5).
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.coef = []

    def soft_threshold(self, x, threshold):
        """
        Perform soft thresholding.

        Parameters:
            x (float): Input value.
            threshold (float): Threshold value.

        Returns:
            float: Soft thresholded value.
        """
        if x > threshold:
            return x - threshold
        elif x < -threshold:
            return x + threshold
        else:
            return 0

    def fit(self, X, y, max_iterations=100, tol=1e-4):
        """
        Fit an Elastic Net regression model to the input data.

        Parameters:
            X (numpy.ndarray): Input feature matrix.
            y (numpy.ndarray): Target values.
            max_iterations (int): Maximum number of iterations for coordinate descent.
            tol (float): Tolerance for convergence.

        Returns:
            None
        """
        n_samples, n_features = X.shape
        self.coef = np.zeros(n_features)
        XTX = X.T.dot(X)

        for _ in range(max_iterations):
            old_coef = np.copy(self.coef)
            for j in range(n_features):
                X_j = X[:, j]
                y_pred = X.dot(self.coef)
                r_j = X_j.T.dot(y - y_pred + self.coef[j] * X_j)
                l1_update = self.soft_threshold(r_j, self.alpha * self.l1_ratio)
                l2_update = self.coef[j] * (1 - self.alpha * (1 - self.l1_ratio))
                self.coef[j] = (l1_update + l2_update) / (XTX[j, j] + self.alpha * (1 - self.l1_ratio))

            if np.linalg.norm(self.coef - old_coef) < tol:
                break

    def predict(self, X):
        """
        Make predictions using the fitted Elastic Net regression model.

        Parameters:
            X (numpy.ndarray): Input feature matrix for prediction.

        Returns:
            numpy.ndarray: Predicted target values.
        """
        return X.dot(self.coef)


class PolynomialLassoRegression:
    """
    A simple implementation of polynomial Lasso regression using NumPy.

    This class provides methods to fit a polynomial Lasso regression model to input data,
    predict target values, calculate coefficients with L1 regularization, and compute mean squared error.

    Attributes:
        degree (int): Degree of the polynomial.
        alpha (float): Regularization parameter.
        coef (numpy.ndarray): Coefficients of the polynomial Lasso regression model.

    Methods:
        polynomial_features(X): Transform input features into polynomial features.
        fit(X, y): Fit a polynomial Lasso regression model to the input data.
        predict(X): Make predictions using the fitted polynomial Lasso regression model.
        mean_squared_error(y_true, y_pred): Calculate mean squared error between true and predicted values.
    """

    def __init__(self, degree=1, alpha=1):
        """
        Initialize the PolynomialLassoRegression class.

        Parameters:
            degree (int): Degree of the polynomial (default is 1).
            alpha (float): Regularization parameter (default is 1).
        """
        self.degree = degree
        self.alpha = alpha
        self.coef = []

    def polynomial_features(self, X):
        """
        Transform input features into polynomial features.

        Parameters:
            X (numpy.ndarray): Input feature matrix.

        Returns:
            numpy.ndarray: Transformed feature matrix with polynomial terms.
        """
        return np.column_stack([X**i for i in range(1, self.degree + 1)])

    def soft_threshold(self, x, threshold):
        """
        Perform soft thresholding.

        Parameters:
            x (float): Input value.
            threshold (float): Threshold value.

        Returns:
            float: Soft thresholded value.
        """
        if x > threshold:
            return x - threshold
        elif x < -threshold:
            return x + threshold
        else:
            return 0

    def fit(self, X, y, max_iterations=100, tol=1e-4):
        """
        Fit a polynomial Lasso regression model to the input data.

        Parameters:
            X (numpy.ndarray): Input feature matrix.
            y (numpy.ndarray): Target values.
            max_iterations (int): Maximum number of iterations for coordinate descent.
            tol (float): Tolerance for convergence.

        Returns:
            None
        """
        X_poly = self.polynomial_features(X)
        X_b = np.c_[np.ones((X_poly.shape[0], 1)), X_poly]
        n_samples, n_features = X_b.shape

        self.coef = np.zeros(n_features)

        for _ in range(max_iterations):
            old_coef = np.copy(self.coef)
            for j in range(n_features):
                X_j = X_b[:, j]
                y_pred = X_b.dot(self.coef)
                r_j = X_j.T.dot(y - y_pred + self.coef[j] * X_j)
                self.coef[j] = self.soft_threshold(r_j, self.alpha) / (X_j**2).sum()

            if np.linalg.norm(self.coef - old_coef) < tol:
                break

    def predict(self, X):
        """
        Make predictions using the fitted polynomial Lasso regression model.

        Parameters:
            X (numpy.ndarray): Input feature matrix for prediction.

        Returns:
            numpy.ndarray: Predicted target values.
        """
        X_poly = self.polynomial_features(X)
        X_b = np.c_[np.ones((X_poly.shape[0], 1)), X_poly]
        return X_b.dot(self.coef)
    

class SGDRegressor:
    """
    A simple implementation of Stochastic Gradient Descent Regressor for linear regression.
    
    Parameters:
    - max_iter (int): Maximum number of iterations for training.
    - random_state (int): Seed for random number generation.
    - learning_rate (float): Learning rate for gradient descent updates.
    """
    
    def __init__(self, max_iter=100, random_state=42, learning_rate=0.1) -> None:
        """
        Initialize the SGDRegressor.
        """
        self.coef = None
        self.max_iter = max_iter
        self.random_state = random_state
        self.learning_rate = learning_rate
    
    def fit(self, X, y):
        """
        Fit the linear regression model using Stochastic Gradient Descent.
        
        Parameters:
        - X (numpy.ndarray): Input features of shape (n_samples, n_features).
        - y (numpy.ndarray): Target values of shape (n_samples,).
        """
        n_samples, n_features = X.shape
        np.random.seed(self.random_state)
        self.coef = np.random.randn(n_features, 1)
        X_b = np.c_[np.ones((n_samples, 1)), X]
        for _ in range(self.max_iter):
            gradient = (2 / n_samples) * X_b.T.dot(X_b.dot(self.coef) - y)
            self.coef = self.coef - self.learning_rate * gradient

    def predict(self, X):
        """
        Predict target values for new data points.
        
        Parameters:
        - X (numpy.ndarray): Input features of shape (n_samples, n_features).
        
        Returns:
        - predictions (numpy.ndarray): Predicted target values of shape (n_samples,).
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.coef)
    

class LogisticRegression:
    """
    A simple implementation of Logistic Regression using gradient descent.
    
    Parameters:
        learning_rate (float): The learning rate for gradient descent. Default is 0.01.
        num_iterations (int): The number of iterations for gradient descent. Default is 1000.
    """
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        """
        Sigmoid activation function.
        
        Parameters:
            z (float or numpy array): The input to the sigmoid function.
        
        Returns:
            float or numpy array: The output of the sigmoid function.
        """
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """
        Fit the logistic regression model to the given training data.
        
        Parameters:
            X (numpy array): Training data features of shape (num_samples, num_features).
            y (numpy array): Target values of shape (num_samples,).
        """
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        iterations_range = tqdm(range(self.num_iterations), desc="Training", unit="iteration")
        # Gradient descent
        for _ in iterations_range:
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            dw = (1/num_samples) * np.dot(X.T, (y_pred - y))
            db = (1/num_samples) * np.sum(y_pred - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            iterations_range.set_postfix({"Loss": self.calculate_loss(X, y)})

    def calculate_loss(self, X, y):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        epsilon = 1e-15  # Small value to prevent log(0)
        loss = -np.mean(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))
        return loss


    def predict(self, X):
        """
        Predict the class labels for the given data.
        
        Parameters:
            X (numpy array): Data features of shape (num_samples, num_features).
        
        Returns:
            list: Predicted class labels (0 or 1) for each sample.
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]
        return y_pred_class