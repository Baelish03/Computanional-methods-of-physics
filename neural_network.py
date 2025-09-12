import numpy as np

class NeuralNetwork:
    """
    Fully connected feedforward neural network with one hidden layer.

    Implements forward propagation, backpropagation, parameter updates,
    and training using gradient descent for multiclass classification.
    """
    def __init__(self, features_matrix, labels_vector,
                 learning_rate, epochs):
        """
        Initialize the neural network.

        Parameters
        ----------
        features_matrix : np.ndarray
            Input features with shape (n_samples, n_features).
        labels_vector : np.ndarray
            Target labels with integer encoding, shape (n_samples,).
        learning_rate : float
            Learning rate for gradient descent.
        epochs : int
            Number of training iterations.
        """
        self.features_matrix = features_matrix
        self.features_len = features_matrix.shape[1]
        self.labels_vector = labels_vector
        hidden_neurons = 250
        labels_len = len(set(labels_vector))
        self.weigth1 = np.random.rand(hidden_neurons, features_matrix.shape[1]) - 0.5
        self.offset1 = np.random.rand(hidden_neurons, 1) - 0.5
        self.weigth2 = np.random.rand(labels_len, hidden_neurons) - 0.5
        self.offset2 = np.random.rand(labels_len, 1) - 0.5
        self.input1, self.input2, self.output1, self.output2 = [], [], [], []
        self.dinput1, self.dinput2, self.doffset1, self.doffset2= [], [], [], []
        self.one_hot_y, self.dweigth1, self.dweigth2 = [], [], []
        self.learning_rate = learning_rate
        self.epochs = epochs

    def sigmoid(self, inp):
        """
        Sigmoid activation function.

        Parameters
        ----------
        inp : np.ndarray
            Input values.

        Returns
        -------
        np.ndarray
            Output after applying the sigmoid function.
        """
        return 1 / (1 + np.e ** (-inp))

    def sigmoid_deriv(self, inp):
        """
        Derivative of the sigmoid activation function.

        Parameters
        ----------
        inp : np.ndarray
            Input values.

        Returns
        -------
        np.ndarray
            Gradient of the sigmoid function evaluated at Z.
        """
        return self.sigmoid(inp) * (1 - self.sigmoid(inp))

    def forward_prop(self, input_matrix):
        """
        Perform forward propagation through the network.

        Parameters
        ----------
        input_matrix : np.ndarray
            Input feature matrix, shape (n_samples, n_features).

        Returns
        -------
        tuple of np.ndarray
            (input1, output1, input2, output2) intermediate and output activations.
        """
        input_matrix = input_matrix.T
        self.input1 = self.weigth1.dot(input_matrix) + self.offset1
        self.output1 = self.sigmoid(self.input1)
        self.input2 = self.weigth2.dot(self.output1) + self.offset2
        self.output2 = self.sigmoid(self.input2)
        return self.input1, self.output1, self.input2, self.output2

    def one_hot(self):
        """
        Convert integer labels to one-hot encoding.

        Returns
        -------
        np.ndarray
            One-hot encoded labels with shape (n_classes, n_samples).
        """
        labels_vector = self.labels_vector
        self.one_hot_y = np.zeros((labels_vector.size, labels_vector.max()+1))
        self.one_hot_y[np.arange(labels_vector.size), labels_vector] = 1
        self.one_hot_y = self.one_hot_y.T
        return self.one_hot_y

    def backward_prop(self):
        """
        Perform backpropagation to compute gradients.

        Returns
        -------
        tuple of np.ndarray
            Gradients (dweigth1, doffset1, dweigth2, doffset2).
        """
        features_len = self.features_len
        self.one_hot_y = self.one_hot()
        self.dinput2 = self.output2 - self.one_hot_y
        self.dweigth2 = 1 / features_len * self.dinput2.dot(self.output1.T)
        self.doffset2 = 1 / features_len * np.sum(self.dinput2)
        self.dinput1 = self.weigth2.T.dot(self.dinput2) * self.sigmoid_deriv(self.input1)
        self.dweigth1 = 1 / features_len * self.dinput1.dot(self.features_matrix)
        self.doffset1 = 1 / features_len * np.sum(self.dinput1)
        return self.dweigth1, self.doffset1, self.dweigth2, self.doffset2

    def update_params(self):
        """
        Update the network parameters using computed gradients.

        Returns
        -------
        tuple of np.ndarray
            Updated parameters (weigth1, offset1, weigth2, offset2).
        """
        learning_rate = self.learning_rate
        self.weigth1 = self.weigth1 - learning_rate * self.dweigth1
        self.offset1 = self.offset1 - learning_rate * self.doffset1
        self.weigth2 = self.weigth2 - learning_rate * self.dweigth2
        self.offset2 = self.offset2 - learning_rate * self.doffset2
        return self.weigth1, self.offset1, self.weigth2, self.offset2

    def gradient_descent(self):
        """
        Train the network using gradient descent.

        Returns
        -------
        tuple of np.ndarray
            Final trained parameters (weigth1, offset1, weigth2, offset2).
        """
        for _ in range(self.epochs):
            self.input1, self.output1, self.input2, self.output2 = self.forward_prop(
                self.features_matrix)
            self.dweigth1, self.doffset1, self.dweigth2, self.doffset2 = self.backward_prop()
            self.weigth1, self.offset1, self.weigth2, self.offset2 = self.update_params()
        return self.weigth1, self.offset1, self.weigth2, self.offset2

    def prediction(self, test_matrix):
        """
        Predict class labels for input samples.

        Parameters
        ----------
        test_matrix : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted class indices, shape (n_samples,).
        """
        _, _, _, output2 = self.forward_prop(test_matrix)
        return np.argmax(output2, 0)

    def score(self):
        """
        Compute training accuracy of the model.

        Returns
        -------
        float
            Training accuracy in [0, 1].
        """
        return np.sum(self.prediction(self.features_matrix) ==
                      self.labels_vector) / self.labels_vector.size

    def one_vs_all(self, test_matrix):
        """
        Dummy method for compatibility with one-vs-all plots.

        Parameters
        ----------
        test_matrix : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted class indices, shape (n_samples,).
        """
        return self.prediction(test_matrix)
