import numpy as np
from cvxopt import matrix, solvers
from one_vs_all import OneVsAll

class SupportVectorMachine(OneVsAll):
    def __init__(self, features_matrix, labels_vector,
                 trained_weights_matrix, scores, length,
                 kernel_type):
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
        if self.kernel_type == "linear":
            return np.dot(input1, input2.T)
        elif self.kernel_type == "gaussian":
            return np.exp(-self.gamma * np.linalg.norm(input1[:, np.newaxis] - input2[np.newaxis, :], axis=2) ** 2)
    
    def weights_training(self):
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

        sv = (alpha > 1e-4)
        b = np.mean(self.new_label[sv] - np.dot(self.features_matrix[sv], w))

        return np.concatenate((np.expand_dims(b, axis =0),w))

    def train_prediction(self, trained_weights_vec):
        return -1 * np.sign(np.dot(trained_weights_vec[1:], self.features_matrix.T) + trained_weights_vec[0])
    
    def test_prediction(self, test_matrix, trained_weights_vec):
        return -1 * np.sign(np.dot(trained_weights_vec[1:], test_matrix.T) + trained_weights_vec[0])
                
    def train_score(self, trained_weights_vec):
        return np.mean(self.new_label == self.train_prediction(trained_weights_vec))
    
    def one_vs_all(self, test_matrix):
        return super().one_vs_all(test_matrix)