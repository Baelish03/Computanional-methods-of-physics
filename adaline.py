import numpy as np
import pandas as pd

class Adaline:
    """
    Implementation of the Adaline (Adaptive Linear Neuron) algorithm.

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
        Number of classes.
    weights_vec : np.ndarray
        Current weight vector (bias + feature weights).
    dummylabel_vec : np.ndarray
        Temporary predictions during training.
    gradient_vec : np.ndarray
        Gradient vector used for weight updates.
    classifier_name : str
        Identifier for the classifier ("Adaline").
    indexes : list[int]
        Class indices used in one-vs-all classification.
    predictions : list
        Predictions from each class in one-vs-all classification.
    """
    def __init__(self, features_matrix, labels_vector,
                 trained_weight_matrix, scores,
                 learning_rate, epochs, length):
        """Initialize the Adaline classifier with dataset and hyperparameters."""
        self.features_matrix = features_matrix
        self.samples_len = features_matrix.shape[0]
        features_len = features_matrix.shape[1]
        self.weights_vec = np.zeros(shape=features_len + 1)
        self.dummylabel_vec = np.zeros(shape=self.samples_len)
        self.labels_vector = labels_vector
        self.gradient_vec = np.zeros(shape=features_len)
        self.trained_weights_matrix = trained_weight_matrix
        self.trained_scores = scores
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.length = length
        self.classifier_name = "Adaline"
        self.indexes, self.predictions = [], [None] * length

    def activation_func(self, features_vec):
        """
        Compute activation function.

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

    def error_func (self, dummylabel, label):
        """
        Compute squared error.

        Parameters
        ----------
        dummylabel : float
            Predicted value.
        label : float
            True label.

        Returns
        -------
        float
            Squared error.
        """
        return (label - dummylabel) ** 2

    def gradient(self, features_vec, dummylabel, label):
        """
        Compute gradient contribution for a single sample.

        Parameters
        ----------
        features_vec : np.ndarray
            Input feature vector.
        dummylabel : float
            Predicted value.
        label : float
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
        learning_rate = self.learning_rate
        weights_vec = self.weights_vec
        weights_vec[1:] -= learning_rate * self.gradient_vec
        weights_vec[0] += learning_rate * sum(self.labels_vector - self.dummylabel_vec)
        return weights_vec

    def weights_training(self):
        """
        Train the Adaline model for the specified number of epochs.

        Returns
        -------
        np.ndarray
            Final trained weight vector.
        """
        error = 0
        step = 0
        while step <= self.epochs:
            error = 0
            self.gradient_vec = np.zeros_like(self.gradient_vec)
            for i in range(self.samples_len):
                labels_vector_element = self.labels_vector[i]
                self.dummylabel_vec[i] = self.activation_func(self.features_matrix[i])
                error += self.error_func (self.dummylabel_vec[i], labels_vector_element)
                self.gradient_vec += self.gradient(self.features_matrix[i],
                                                   self.dummylabel_vec[i],
                                                   labels_vector_element)
            self.weights_vec = self.weights_vec_update()
            step += 1
        #print("Number of wrong label:", error)
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
        return np.where(np.dot(trained_weights_vec[1:],
                               self.features_matrix.T) + trained_weights_vec[0] >= 0,
                               1, 0)

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
        return (np.where(np.dot(trained_weights_vec[1:],
                                test_matrix.T) + trained_weights_vec[0] >= 0,
                                1, 0))

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

        Each Adaline corresponds to a single class.
        Predictions are collected from all classifiers and the
        final class is chosen based on scores.

        Parameters
        ----------
        test_matrix : np.ndarray
            Test feature matrix.

        Returns
        -------
        np.ndarray
            Final predicted class labels.
        """
        length = self.length
        self.indexes, self.predictions = [], [None] * length
        for a in range(length):
            self.indexes.append(a)
            self.predictions[a] = self.test_prediction(test_matrix, self.trained_weights_matrix[a])
        df = pd.DataFrame()
        df["indexes"] = self.indexes
        df["scores"] = self.trained_scores
        df["pred"] = self.predictions
        df.sort_values(by="scores", ascending=False, inplace=True, ignore_index=True)
        final_pred = np.full(fill_value=None,shape=np.array(self.predictions).shape[1])
        for i in range(0,len(df)-1):
            for j in range(len(final_pred)):
                if final_pred[j] is None and df["pred"][i][j] == 1:
                    final_pred[j] = df['indexes'][i]
        for i in range(len(final_pred)):
            if final_pred[i] is None:
                final_pred[i] = df['indexes'].iloc[-1]
        return np.array(final_pred.tolist(), dtype=float)
