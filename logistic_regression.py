import numpy as np
from one_vs_all import OneVsAll

class LogisticRegression(OneVsAll):
    """
    Implementation of Logistic Regression classifier.

    Inherits from OneVsAll to support multi-class classification. 
    Uses gradient descent to optimize the log-likelihood function. 
    Provides training, prediction, and scoring utilities.

    Attributes
    ----------
    features_matrix : np.ndarray
        Training features of shape (n_samples, n_features).
    labels_vector : np.ndarray
        Ground-truth labels (0 or 1 for binary classification).
    trained_weights_matrix : np.ndarray
        Pre-trained weight vectors for one-vs-all classification.
    trained_scores : list[float]
        Performance scores of each trained classifier.
    learning_rate : float
        Step size for weight updates.
    epochs : int
        Number of iterations over the dataset.
    length : int
        Number of classes.
    weights_vec : np.ndarray
        Current weight vector (bias + feature weights).
    dummylabel_vec : np.ndarray
        Temporary predictions during training.
    gradient_vec : np.ndarray
        Gradient vector used for weight updates.
    p_vec : np.ndarray
        Vector of probabilities (sigmoid outputs).
    classifier_name : str
        Identifier for the classifier ("Logistic regression").
    """
    def __init__(self, features_matrix, labels_vector,
                 trained_weights_matrix, scores,
                 learning_rate, epochs, length):
        """Initialize the Logistic Regression classifier with dataset and hyperparameters."""
        self.features_matrix = features_matrix
        self.samples_len = features_matrix.shape[0]
        features_len = features_matrix.shape[1]
        self.weights_vec = np.zeros(features_len + 1)
        self.dummylabel_vec = np.zeros(shape=self.samples_len)
        self.labels_vector = labels_vector
        self.gradient_vec = np.zeros(shape=features_len)
        self.p_vec = np.zeros(shape=self.samples_len)
        self.trained_scores = scores
        self.trained_weights_matrix = trained_weights_matrix
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.length = length
        self.classifier_name = "Logistic regression"

    def sigmoid(self, test_vec, weights_vec):
        """
        Compute the sigmoid function.

        Parameters
        ----------
        test_vec : np.ndarray
            Input feature vector.
        weights_vec : np.ndarray
            Current weight vector.

        Returns
        -------
        float
            Probability value in (0, 1).
        """
        z = np.dot(test_vec, weights_vec[1:]) + weights_vec[0]
        return 1 / (1 + np.e ** (-z))

    def activation_func(self, p):
        """
        Apply threshold to sigmoid probability.

        Parameters
        ----------
        p : float
            Probability value in (0, 1).

        Returns
        -------
        int
            Predicted label (1 if p > 0.5, else 0).
        """
        return np.where(p > 0.5, 1, 0)

    def gradient(self, features_vec, dummylabel, label):
        """
        Compute gradient contribution for a single sample.

        Parameters
        ----------
        features_vec : np.ndarray
            Input feature vector.
        dummylabel : int
            Predicted label (0 or 1).
        label : int
            True label.

        Returns
        -------
        np.ndarray
            Gradient vector for this sample.
        """
        return (label - dummylabel) * (-1) * features_vec

    def weights_vec_update(self):
        """
        Update the weight vector using accumulated gradients.

        Returns
        -------
        np.ndarray
            Updated weight vector.
        """
        weights_vec = self.weights_vec
        learning_rate = self.learning_rate
        weights_vec[1:] -= learning_rate * self.gradient_vec
        weights_vec[0] += learning_rate * sum(self.labels_vector - self.dummylabel_vec)
        return weights_vec

    def weights_training(self):
        """
        Train the Logistic Regression model using gradient descent.

        Returns
        -------
        np.ndarray
            Final trained weight vector.
        """
        step = 0
        while step <= self.epochs:
            self.gradient_vec = np.zeros_like(self.gradient_vec)
            for i in range(self.samples_len):
                self.p_vec[i] = self.sigmoid(self.features_matrix[i], self.weights_vec)
                self.dummylabel_vec[i] = self.activation_func(self.p_vec[i])
                self.gradient_vec += self.gradient(self.features_matrix[i],
                                                   self.dummylabel_vec[i],
                                                   self.labels_vector[i])
            self.weights_vec = self.weights_vec_update()
            step += 1
        return self.weights_vec

    def train_prediction(self, trained_weights_vec):
        """
        Predict labels for the training set using given weights.

        Parameters
        ----------
        trained_weights_vec : np.ndarray
            Trained weight vector.

        Returns
        -------
        np.ndarray
            Array of predicted labels (0 or 1).
        """
        z = np.dot(trained_weights_vec[1:], self.features_matrix.T) + trained_weights_vec[0]
        return (np.where(1 / (1 + np.e ** (-z)) > 0.5, 1, 0))

    def test_prediction(self, test_matrix, trained_weights_vec):
        """
        Predict labels for a test dataset using given weights.

        Parameters
        ----------
        test_matrix : np.ndarray
            Test feature matrix.
        trained_weights_vec : np.ndarray
            Trained weight vector.

        Returns
        -------
        np.ndarray
            Array of predicted labels (0 or 1).
        """
        z = np.dot(trained_weights_vec[1:], test_matrix.T) + trained_weights_vec[0]
        return (np.where(1 / (1 + np.e ** (-z)) > 0.5, 1, 0))

    def train_score(self, trained_weights_vec):
        """
        Compute training accuracy.

        Parameters
        ----------
        trained_weights_vec : np.ndarray
            Trained weight vector.

        Returns
        -------
        float
            Accuracy score in [0, 1].
        """
        return np.mean(self.labels_vector == self.train_prediction(trained_weights_vec))

    def one_vs_all(self, test_matrix):
        """
        Multi-class classification using one-vs-all strategy.

        Parameters
        ----------
        test_matrix : np.ndarray
            Test feature matrix.

        Returns
        -------
        np.ndarray
            Final predicted class labels.
        """
        return super().one_vs_all(test_matrix)
