import sys
import statistics as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from perceptron import Perceptron
from adaline import Adaline
from logistic_regression import LogisticRegression
from support_vector_machine import SupportVectorMachine
from neural_network import NeuralNetwork

np.random.seed(1)

def normalization(features_matrix):
    """
    Standardize each feature column by subtracting its mean and dividing
    by its standard deviation.

    Parameters
    ----------
    features_matrix : np.ndarray
        Input feature matrix, shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
        Normalized feature matrix with zero mean and unit variance
        per column.
    """
    norm_features_matrix = np.zeros(shape=features_matrix.shape)
    for i in range(features_matrix.shape[1]):
        col_mean = st.mean(features_matrix[:,i])
        col_stdev = st.stdev(features_matrix[:,i])
        norm_features_matrix[:,i] = (features_matrix[:,i] - col_mean) / col_stdev
    return norm_features_matrix

def one_vs_all_label(labels_vec):
    """
    Convert class labels into one-vs-all binary label vectors.

    Parameters
    ----------
    labels_vec : np.ndarray
        Target labels, shape (n_samples,).

    Returns
    -------
    list of list of int
        A list of binary label vectors, one per class.
    """
    target_list = list(set(labels_vec))
    target_matrix=[]
    for i in range(len(target_list)):
        single_target_vec = list(np.where(labels_vec == target_list[i], 1, 0))
        target_matrix.append(single_target_vec)
    return target_matrix

def plot(classifier, title=''):
    """
    Visualize decision boundaries and data points for a classifier.

    Parameters
    ----------
    classifier : object
        A trained classifier with a one_vs_all method.
    title : str, optional
        Title for the plot.
    """
    test_idx = range(len(TEST_MATRIX))
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(GLOBAL_LABELS))])
    for idx, cl in enumerate(np.unique(GLOBAL_LABELS)):
        plt.scatter(x=GLOBAL_MATRIX[GLOBAL_LABELS == cl, 0],
                    y=GLOBAL_MATRIX[GLOBAL_LABELS == cl, 1],
                    alpha=0.8, c=colors[idx], marker=markers[idx],
                    label=f'Class {cl}', edgecolor='black')
    if test_idx:
        x_test, _ = TEST_MATRIX[test_idx, :], GLOBAL_LABELS[test_idx]

        plt.scatter(x_test[:, 0], x_test[:, 1], c='none', edgecolor='black',
                    alpha=1.0, linewidth=1, marker='o', s=100, label='Test set')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx_grid = np.linspace(xlim[0], xlim[1], 200)
    yy_grid = np.linspace(ylim[0], ylim[1], 200)
    yy_grid, xx_grid = np.meshgrid(yy_grid, xx_grid)
    xy_grid = np.vstack([xx_grid.ravel(), yy_grid.ravel()]).T
    z_value = classifier.one_vs_all(xy_grid).reshape(xx_grid.shape)
    ax.contourf(xx_grid, yy_grid, z_value, alpha=0.3, cmap=cmap)
    plt.title(title)
    plt.xlabel('Petal length [standardized]')
    plt.ylabel('Petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def analysis(labels_vector, features_matrix, test_matrix, test_labels,
             classifier):
    """
    Train and evaluate a one-vs-all classifier.

    Parameters
    ----------
    labels_vector : np.ndarray
        Class labels for training samples.
    features_matrix : np.ndarray
        Training feature matrix.
    test_matrix : np.ndarray
        Feature matrix for testing.
    test_labels : np.ndarray
        True labels for the test set.
    classifier : class
        Classifier class to instantiate and train.

    Returns
    -------
    object
        Trained classifier instance.
    """
    labels_matrix = one_vs_all_label(labels_vector)
    length = len(labels_matrix)
    trained_weights_matrix = [None] * length
    scores = [None] * length
    for a in range(length):
        labels_vector = labels_matrix[a]
        classifier_var = classifier(features_matrix, labels_vector,
                                    trained_weights_matrix, scores,
                                    LEARNING_RATE, EPOCHS, length)
        trained_weights_matrix[a] = list(classifier_var.weights_training())
        scores[a] = classifier_var.train_score(trained_weights_matrix[a])
    #predicted_train_label = classifier_var.one_vs_all(features_matrix)
    predicted_test_label = classifier_var.one_vs_all(test_matrix)
    print('Misclassified Test Samples: %d'
          % (predicted_test_label != test_labels).sum(),
          " / ", len(test_labels))
    plot(classifier_var, title=classifier_var.classifier_name)
    return classifier_var

def svm_analysis(labels_vector, features_matrix, test_matrix,
                 test_labels, kernel_type):
    """
    Train and evaluate a Support Vector Machine classifier.

    Parameters
    ----------
    labels_vector : np.ndarray
        Class labels for training samples.
    features_matrix : np.ndarray
        Training feature matrix.
    test_matrix : np.ndarray
        Feature matrix for testing.
    test_labels : np.ndarray
        True labels for the test set.
    kernel_type : str
        Kernel type for SVM, e.g. "linear" or "gaussian".

    Returns
    -------
    object
        Trained SupportVectorMachine instance.
    """
    labels_matrix = one_vs_all_label(labels_vector)
    length = len(labels_matrix)
    trained_weights_matrix = [None] * length
    scores = [None] * length
    for a in range(length):
        labels_vector = labels_matrix[a]
        classifier_var = SupportVectorMachine(features_matrix, labels_vector,
                                              trained_weights_matrix, scores,
                                              length, kernel_type)
        trained_weights_matrix[a] = list(classifier_var.weights_training())
        scores[a] = classifier_var.train_score(trained_weights_matrix[a])
    #PREDICTED_TRAIN_LABEL = classifier_var.one_vs_all(features_matrix)
    predicted_test_label = classifier_var.one_vs_all(test_matrix)
    print('Misclassified Test Samples: %d'
          % (predicted_test_label != test_labels).sum(),
          " / ", len(test_labels))
    plot(classifier_var, title=(classifier_var.classifier_name +
                                " with " + kernel_type + " kernel"))

if __name__ == "__main__":
    DATASET = load_iris(as_frame=True)
    GLOBAL_MATRIX = normalization( DATASET.data.values [:,[2,3]])
    GLOBAL_LABELS = load_iris().target
    FEATURES_MATRIX, TEST_MATRIX, LABELS_VECTOR, TEST_LABELS = (
        train_test_split(GLOBAL_MATRIX, GLOBAL_LABELS,
                         test_size=0.3, random_state=1,
                         stratify=GLOBAL_LABELS))
    EPOCHS = 1000
    LEARNING_RATE = 0.001

    ASK = str(input("""1. Perceptron
  2. Adaline
  3. Logistic Regression
  4. Support Vector Machine (linear kernel)
  5. Support Vector Machine (gaussian kernel)
  6. Neural Network
  Select a classifier: """, ))

    if ASK == "1":
        analysis(LABELS_VECTOR, FEATURES_MATRIX,
                                  TEST_MATRIX, TEST_LABELS, Perceptron)

    elif ASK == "2":
        analysis(LABELS_VECTOR, FEATURES_MATRIX,
                                  TEST_MATRIX, TEST_LABELS, Adaline)

    elif ASK == "3":
        analysis(LABELS_VECTOR, FEATURES_MATRIX,
                                  TEST_MATRIX, TEST_LABELS, LogisticRegression)

    elif ASK == "4":
        svm_analysis(LABELS_VECTOR, FEATURES_MATRIX,
                                      TEST_MATRIX, TEST_LABELS, "linear")

    elif ASK == "5":
        svm_analysis(LABELS_VECTOR, FEATURES_MATRIX,
                                      TEST_MATRIX, TEST_LABELS, "gaussian")

    elif ASK == "6":
        classifier_var = NeuralNetwork(FEATURES_MATRIX, LABELS_VECTOR,
                                       LEARNING_RATE, EPOCHS)
        classifier_var.gradient_descent()
        predicted_test_label = classifier_var.prediction(TEST_MATRIX)
        print('Misclassified Test Samples:'
                '%d' % (predicted_test_label != TEST_LABELS).sum(),
                " / ", len(TEST_LABELS))
        plot(classifier_var, title="Neural network")

    else:
        print("Invalid input. Bye bye")
        sys.exit()
