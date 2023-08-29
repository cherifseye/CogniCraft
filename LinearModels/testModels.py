import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class LogisticRegressionWithL2:
    """
    Logistic Regression with L2 regularization using gradient descent.
    
    Parameters:
        learning_rate (float): The learning rate for gradient descent. Default is 0.01.
        num_iterations (int): The number of iterations for gradient descent. Default is 1000.
        lambda_param (float): The regularization parameter for L2 regularization. Default is 0.01.
    """
    def __init__(self, learning_rate=0.01, num_iterations=1000, lambda_param=0.01):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        iterations_range = tqdm(range(self.num_iterations), desc="Training", unit="iteration")
        
        for _ in iterations_range:
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            
            # Compute gradients with L2 regularization
            dw = (1/num_samples) * np.dot(X.T, (y_pred - y)) + (2 * self.lambda_param * self.weights)
            db = (1/num_samples) * np.sum(y_pred - y)
            
            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            iterations_range.set_postfix({"Loss": self.calculate_loss(X, y)})

    def calculate_loss(self, X, y):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        epsilon = 1e-15
        loss = -np.mean(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))
        # Add L2 regularization term to the loss
        regularization_term = (self.lambda_param / (2 * len(y))) * np.sum(self.weights**2)
        loss += regularization_term
        return loss

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]
        return y_pred_class
    
mnist = fetch_openml('mnist_784', as_frame=False)
X, y = mnist["data"], mnist["target"]

X_train, y_train, X_test, y_test = X[:60000], y[:60000], X[60000:], y[60000:]
y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')

sgd_reg = LogisticRegressionWithL2()
sgd_reg.fit(X_train, y_train_5)


y_pred_5 = sgd_reg.predict(X_train)

print(confusion_matrix(y_train_5, y_pred_5))

precision = precision_score(y_train_5, y_pred_5)
recall    = recall_score(y_train_5, y_pred_5)
print(precision, recall) #About 89% of precision