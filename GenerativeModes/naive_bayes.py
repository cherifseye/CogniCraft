import numpy as np
from collections import defaultdict

class DirichletMultinomialClassifier:
    """
    A simple implementation of the Dirichlet Multinomial Classifier for text classification.

    Parameters:
        alpha (float, optional): The Dirichlet prior hyperparameter for Laplace smoothing. Default is 0.1.
    """

    def __init__(self, alpha=0.1):
        """
        Initializes the Dirichlet Multinomial Classifier with the specified alpha.

        Args:
            alpha (float, optional): The Dirichlet prior hyperparameter for Laplace smoothing. Default is 0.1.
        """
        self.alpha = alpha  
        self.class_probs = None  
        self.word_probs = None

    def fit(self, X, y):
        """
        Fits the classifier to the training data.

        Args:
            X (numpy.ndarray): The document-term matrix representing the training data.
            y (numpy.ndarray): The class labels corresponding to the training data.
        """
 
        num_classes = np.max(y) + 1
        num_words = X.shape[1]
    
        self.class_probs = np.zeros(num_classes)
        self.word_probs = np.zeros((num_classes, num_words))
    
        for c in range(num_classes):
            class_mask = (y == c)
            class_count = np.sum(class_mask)
            self.class_probs[c] = (class_count + self.alpha) / (len(y) + num_classes * self.alpha)
    
            for j in range(num_words):
                word_count = np.sum(X[class_mask, j])
                total_word_count = np.sum(X[class_mask, :])
                
                if total_word_count > 0:
                    self.word_probs[c, j] = (word_count + self.alpha) / (total_word_count + num_words * self.alpha)
                else:
                    self.word_probs[c, j] = self.alpha / (total_word_count + num_words * self.alpha)
    
    def predict(self, X):
        """
        Predicts class labels for the input data.

        Args:
            X (numpy.ndarray): The document-term matrix representing the input data.

        Returns:
            numpy.ndarray: Predicted class labels for the input data.
        """
        num_samples, num_words = X.shape
        num_classes = len(self.class_probs)
        predictions = np.zeros(num_samples, dtype=int)

        for i in range(num_samples):
            sample_probs = np.zeros(num_classes)

            for c in range(num_classes):
                class_prob = np.log(self.class_probs[c])

                for j in range(num_words):
                    word_count = X[i, j]
                    word_prob = self.word_probs[c, j]
                    if word_count > 0:
                        class_prob += word_count * np.log(word_prob)

                sample_probs[c] = class_prob

            predictions[i] = np.argmax(sample_probs)

        return predictions


class NaiveBayesClassifier:
    """
    A simple implementation of the Naive Bayes Classifier for classification tasks.

    Attributes:
        class_probs (dict): Dictionary to store class probabilities.
        feature_probs (defaultdict): Default dictionary to store conditional feature probabilities.
        classes (numpy.ndarray): Array to store unique class labels.
    """

    def __init__(self):
        """
        Initializes the Naive Bayes Classifier.
        """
        self.class_probs = {}
        self.feature_probs = defaultdict(lambda: defaultdict(dict))

    def fit(self, X, y):
        """
        Fits the classifier to the training data.

        Args:
            X (numpy.ndarray): The feature matrix representing the training data.
            y (numpy.ndarray): The class labels corresponding to the training data.
        """
        num_samples, num_features = X.shape
        self.classes = np.unique(y)

        for c in self.classes:
            c_mask = (y == c)
            self.class_probs[c] = np.sum(c_mask) / num_samples

            for feature in range(num_features):
                feature_values = X[c_mask][:, feature]
                unique_values, counts = np.unique(feature_values, return_counts=True)
                total_counts = len(feature_values)
                prob_dict = dict(zip(unique_values, counts / total_counts))
                self.feature_probs[c][feature] = prob_dict

    def predict(self, X):
        """
        Predicts class labels for the input data.

        Args:
            X (numpy.ndarray): The feature matrix representing the input data.

        Returns:
            list: Predicted class labels for the input data.
        """
        num_samples, num_features = X.shape
        predictions = []

        for i in range(num_samples):
            sample_probs = {}

            for c in self.classes:
                class_prob = np.log(self.class_probs[c])

                for feature in range(num_features):
                    feature_value = X[i, feature]
                    if feature_value in self.feature_probs[c][feature]:
                        class_prob += np.log(self.feature_probs[c][feature][feature_value])
                    else:
                        class_prob += np.log(1e-10)  # Smoothing for unseen feature values

                sample_probs[c] = class_prob

            predicted_class = max(sample_probs, key=sample_probs.get)
            predictions.append(predicted_class)

        return predictions
