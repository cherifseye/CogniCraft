import numpy as np

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



# Example usage
if __name__ == "__main__":
    np.random.seed(0)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + 1.5 * X**2 + np.random.randn(100, 1)

    poly_regression = PolynomialRegression(degree=2)
    poly_regression.fit(X, y)

    new_X = np.linspace(0, 2, 100).reshape(-1, 1)
    predicted_y = poly_regression.predict(new_X)

    print("Coefficients:", poly_regression.coef)
