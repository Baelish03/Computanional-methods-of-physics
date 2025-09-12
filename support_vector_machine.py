import numpy as np
from cvxopt import matrix, solvers
from one_vs_all import OneVsAll

class SupportVectorMachine(OneVsAll):
    """
    Support Vector Machine (SVM) classifier with linear and Gaussian kernels.

    This class implements a binary Support Vector Machine using quadratic 
    programming from `cvxopt`. It can be extended to multiclass 
    classification via the `OneVsAll` strategy.
    """
    def __init__(self, features_matrix, labels_vector,
                 trained_weights_matrix, scores, length,
                 kernel_type):
        """
        Initialize the Support Vector Machine.

        Parameters
        ----------
        features_matrix : np.ndarray
            Matrix of input features with shape (n_samples, n_features).
        labels_vector : np.ndarray
            Target labels, must contain class `1` and others mapped to `-1`.
        trained_weights_matrix : np.ndarray
            Matrix to store trained weight vectors for one-vs-all classification.
        scores : list
            Accuracy scores for different classifiers in one-vs-all setup.
        length : int
            Number of unique labels (used in one-vs-all strategy).
        kernel_type : str
            Type of kernel function ("linear" or "gaussian").
        """
        self.new_label = np.where(np.array(labels_vector) == 1, 1, -1)
        self.features_matrix = features_matrix
        self.samples_len = features_matrix.shape[0]
        self.alpha = np.zeros(features_matrix.shape[0])
        self.gamma = .1
        self.trained_scores = scores
        self.trained_weights_matrix = trained_weights_matrix
        self.length = length
        self.kernel_type = kernel_type
        self.classifier_name = "Support vector machine"

    def kernel(self, input1, input2):
        """
        Compute the kernel matrix between two sets of vectors.

        Parameters
        ----------
        input1 : np.ndarray
            First set of vectors with shape (n_samples1, n_features).
        input2 : np.ndarray
            Second set of vectors with shape (n_samples2, n_features).

        Returns
        -------
        np.ndarray
            Kernel matrix of shape (n_samples1, n_samples2).
        """
        if self.kernel_type == "linear":
            return np.dot(input1, input2.T)
        if self.kernel_type == "gaussian":
            return np.exp(-self.gamma * np.linalg.norm(input1[:, np.newaxis]
                                                       - input2[np.newaxis, :], axis=2) ** 2)
        return None

    def weights_training(self):
        """
        Train the SVM model using quadratic programming.

        Solves the dual optimization problem and computes the optimal 
        separating hyperplane.

        Returns
        -------
        np.ndarray
            Weight vector including bias term, shape (n_features + 1,).
        """
        samples_len = self.samples_len
        features_matrix = self.features_matrix
        alpha = self.alpha
        K = self.kernel(features_matrix, features_matrix)
        P = matrix(np.outer(self.new_label, self.new_label) * K)
        q = matrix(-np.ones(samples_len))
        G = matrix(np.vstack((-np.eye(samples_len), np.eye(samples_len))))
        h = matrix(np.hstack((np.zeros(samples_len), np.ones(samples_len) * 1.0)))
        A = matrix(self.new_label, (1, samples_len), 'd')
        b = matrix(0.0)
        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)
        alpha = np.array(solution['x']).ravel()
        w = np.sum(np.dot(alpha * self.new_label[:, None], self.features_matrix), axis=0)
        sv = alpha > 1e-4
        b = np.mean(self.new_label[sv] - np.dot(self.features_matrix[sv], w))
        return np.concatenate((np.expand_dims(b, axis =0),w))

    def train_prediction(self, trained_weights_vec):
        """
        Predict labels for the training set.

        Parameters
        ----------
        trained_weights_vec : np.ndarray
            Trained weight vector including bias term.

        Returns
        -------
        np.ndarray
            Predicted labels for training samples, values in {-1, 1}.
        """
        return -1 * np.sign(np.dot(trained_weights_vec[1:],
                                   self.features_matrix.T) + trained_weights_vec[0])

    def test_prediction(self, test_matrix, trained_weights_vec):
        """
        Predict labels for unseen test samples.

        Parameters
        ----------
        test_matrix : np.ndarray
            Matrix of test features with shape (n_samples, n_features).
        trained_weights_vec : np.ndarray
            Trained weight vector including bias term.

        Returns
        -------
        np.ndarray
            Predicted labels for test samples, values in {-1, 1}.
        """
        return -1 * np.sign(np.dot(trained_weights_vec[1:], test_matrix.T) + trained_weights_vec[0])

    def train_score(self, trained_weights_vec):
        """
        Compute training accuracy score.

        Parameters
        ----------
        trained_weights_vec : np.ndarray
            Trained weight vector including bias term.

        Returns
        -------
        float
            Training accuracy as a fraction in [0, 1].
        """
        return np.mean(self.new_label == self.train_prediction(trained_weights_vec))

    def one_vs_all(self, test_matrix):
        """
        Perform one-vs-all classification for multiclass problems.

        Parameters
        ----------
        test_matrix : np.ndarray
            Matrix of test features with shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted class labels for the test set.
        """
        return super().one_vs_all(test_matrix)
