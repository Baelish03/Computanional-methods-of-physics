import numpy as np
from one_vs_all import OneVsAll

class Perceptron(OneVsAll):
    """
    Implementation of the classic Perceptron learning algorithm.

    Supports binary classification and one-vs-all strategy for multi-class
    classification. Provides training, prediction, and scoring utilities.

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
        Number of classes (note: "length" is a typo, should be "length").
    weights_vec : np.ndarray
        Current weight vector (bias + feature weights).
    step : int
        Training step counter.
    classifier_name : str
        Identifier for the classifier ("Perceptron").
    indexes : list[int]
        Class indices used in one-vs-all classification.
    predictions : list
        Predictions from each class in one-vs-all classification.
    """
    def __init__(self, features_matrix, labels_vector,
                 trained_weights_matrix, scores,
                 learning_rate, epochs, length):
        """
        Initialize the perceptron with dataset and hyperparameters.
        """
        self.features_matrix = features_matrix
        self.samples_len = features_matrix.shape[0]
        self.weights_vec = np.zeros(shape=features_matrix.shape[1] + 1)
        self.labels_vector = labels_vector
        self.trained_weights_matrix = trained_weights_matrix
        self.trained_scores = scores
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.length = length
        self.classifier_name = "Perceptron"
        self.indexes, self.predictions = [], [None] * length

    def activation_func(self, features_vec):
        """
        Compute perceptron activation.

        Parameters
        ----------
        features_vec : np.ndarray
            Input feature vector.

        Returns
        -------
        int
            Predicted label (1 if linear combination >= 0, else 0).
        """
        weights_vec = self.weights_vec
        return np.where(np.dot(weights_vec[1:], features_vec) + weights_vec[0] >= 0, 1, 0)

    def weights_vec_update(self, features_vec, dummylabel, label):
        """
        Update the weight vector using the perceptron learning rule:
        w <- w + η * (y - ŷ) * x
        b <- b + η * (y - ŷ)

        Parameters
        ----------
        features_vec : np.ndarray
            Input feature vector.
        dummylabel : int
            Current prediction (ŷ).
        label : int
            True label (y).

        Returns
        -------
        np.ndarray
            Updated weight vector.
        """
        learning_rate = self.learning_rate
        weights_vec = self.weights_vec
        weights_vec[0] += learning_rate * (label - dummylabel)
        weights_vec[1:] += learning_rate * (label - dummylabel) * features_vec
        return weights_vec

    def weights_training(self):
        """
        Train the perceptron for the specified number of epochs.

        Returns
        -------
        np.ndarray
            Final trained weight vector.
        """
        dummylabel_vec = np.zeros(shape=self.samples_len)
        step = 0
        while step <= self.epochs:
            for i in range(self.samples_len):
                feature_matrix_element = self.features_matrix[i]
                dummylabel_vec[i] = self.activation_func(feature_matrix_element)
                self.weights_vec = self.weights_vec_update(feature_matrix_element,
                                                           dummylabel_vec[i],
                                                           self.labels_vector[i])
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
        return (np.where(np.dot(trained_weights_vec[1:],
                                self.features_matrix.T) + trained_weights_vec[0] >= 0, 1, 0))

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
        return (np.where(np.dot(trained_weights_vec[1:], test_matrix.T)
                         + trained_weights_vec[0] >= 0, 1, 0))

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
