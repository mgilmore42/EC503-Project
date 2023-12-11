from collections import OrderedDict
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from pprint import pprint

from sklearn.model_selection import train_test_split

from utils.data import HeartDataset, HousingDataset, RainDataset, CampusDataset
import time
import polars as pl

def train_gradient_boosting_classifier(X, y, n_estimators=100):
    """
    Train a Gradient Boosting Classifier.

    Parameters:
    X: array-like, shape = [n_samples, n_features]
       Training vectors, where n_samples is the number of samples and
       n_features is the number of features.
    y: array-like, shape = [n_samples]
       Target values (class labels in classification).

    Returns:
    model: The trained Gradient Boosting Classifier model.
    """

    # Create a Gradient Boosting Classifier
    gbc = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=1e0, max_depth=1, random_state=42)

    # Train the classifier
    gbc.fit(X, y)

    return gbc

def train_logistic_regression(X, y):
    """
    Train a Logistic Regression Classifier.

    Parameters:
    X: array-like, shape = [n_samples, n_features]
       Training vectors, where n_samples is the number of samples and
       n_features is the number of features.
    y: array-like, shape = [n_samples]
       Target values (class labels in classification).

    Returns:
    model: The trained Logistic Regression Classifier model.
    """

    # Create a Logistic Regression Classifier
    lrc = LogisticRegression(max_iter=2_000, random_state=42)

    # Train the classifier
    lrc.fit(X, y)

    return lrc

def train_random_forest_classifier(X, y, n_estimators=100):
    """
    Train a Random Forest Classifier.

    Parameters:
    X: array-like, shape = [n_samples, n_features]
       Training vectors, where n_samples is the number of samples and
       n_features is the number of features.
    y: array-like, shape = [n_samples]
       Target values (class labels in classification).

    Returns:
    model: The trained Random Forest Classifier model.
    """

    # Create a Random Forest Classifier
    rfc = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

    # Train the classifier
    rfc.fit(X, y)

    return rfc

def train_support_vector_classifier(X, y, kernel):
    """
    Train a Support Vector Classifier.

    Parameters:
    X: array-like, shape = [n_samples, n_features]
       Training vectors, where n_samples is the number of samples and
       n_features is the number of features.
    y: array-like, shape = [n_samples]
       Target values (class labels in classification).

    Returns:
    model: The trained Support Vector Classifier model.
    """

    # Create a Support Vector Classifier
    svc = SVC(kernel=kernel, gamma='auto', random_state=42)

    # Train the classifier
    svc.fit(X, y)

    return svc

def train_neural_network_classifier(X, y):
    """
    Train a Neural Network Classifier.

    Parameters:
    X: array-like, shape = [n_samples, n_features]
       Training vectors, where n_samples is the number of samples and
       n_features is the number of features.
    y: array-like, shape = [n_samples]
       Target values (class labels in classification).

    Returns:
    model: The trained Neural Network Classifier model.
    """

    # Create a Neural Network Classifier
    nnc = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 5, 5), max_iter=2_000, random_state=42)

    # Train the classifier
    nnc.fit(X, y)

    return nnc

def train_all_classifiers(X, y, n_estimators=100, kernel='rbf'):
    """
    Train all the classifiers on the given dataset.

    Parameters:
    X: array-like, shape = [n_samples, n_features]
       Training vectors, where n_samples is the number of samples and
       n_features is the number of features.
    y: array-like, shape = [n_samples]
       Target values (class labels in classification).

    Returns:
    models: A dictionary of all the trained models.
    """

    training_times = {}

    start_time = time.time()
    gbc = train_gradient_boosting_classifier(X, y, n_estimators=n_estimators)
    training_times['GradientBoosting'] = time.time() - start_time

    start_time = time.time()
    lrc = train_logistic_regression(X, y)
    training_times['LogisticRegression'] = time.time() - start_time

    start_time = time.time()
    rfc = train_random_forest_classifier(X, y, n_estimators=n_estimators)
    training_times['RandomForest'] = time.time() - start_time

    start_time = time.time()
    svc = train_support_vector_classifier(X, y, kernel)
    training_times['SupportVector'] = time.time() - start_time

    start_time = time.time()
    nnc = train_neural_network_classifier(X, y)
    training_times['NeuralNetwork'] = time.time() - start_time

    return gbc, lrc, rfc, svc, nnc, training_times

def get_accuracy(X_train, y_train, X_test, y_test, gbc, lrc, rfc, svc, nnc, training_times):

    metrics = OrderedDict(
        accuracy = ['test', 'train', 'training time'],
        grad_boost = [
            gbc.score(X_test, y_test),
            gbc.score(X_train, y_train),
            training_times['GradientBoosting']
        ],
        log_reg = [
            lrc.score(X_test, y_test),
            lrc.score(X_train, y_train),
            training_times['LogisticRegression']
        ],
        rand_forest = [
            rfc.score(X_test, y_test),
            rfc.score(X_train, y_train),
            training_times['RandomForest']
        ],
        svc = [
            svc.score(X_test, y_test),
            svc.score(X_train, y_train),
            training_times['SupportVector']
        ],
        neural_net = [
            nnc.score(X_test, y_test),
            nnc.score(X_train, y_train),
            training_times['NeuralNetwork']
        ]
    )

    return pl.DataFrame(metrics)

def train_campus():

    dataset = CampusDataset()
    
    target = 'degree_t'
    features = list(set(dataset.df.columns) - set([target, 'salary']))

    y, X = dataset.get(features, target, ommit_nan=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifiers = train_all_classifiers(X_train, y_train, kernel='linear')

    metrics = get_accuracy(X_train, y_train, X_test, y_test, *classifiers)

    print('Campus Placement Dataset')
    print(metrics)
    print()
    print()

def train_heart():
    
    dataset = HeartDataset()

    target = 'output'
    features = list(set(dataset.df.columns) - set([target]))

    y, X = dataset.get(features, target, ommit_nan=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifiers = train_all_classifiers(X_train, y_train, kernel='poly')

    metrics = get_accuracy(X_train, y_train, X_test, y_test, *classifiers)

    print('Heart Disease Dataset')
    print(metrics)
    print()
    print()

def train_rain():

    dataset = RainDataset()

    target = 'RainTomorrow'
    features = list(set(dataset.df.columns) - set([target]))

    y, X = dataset.get(features, target, ommit_nan=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifiers = train_all_classifiers(X_train, y_train, n_estimators=5_000, kernel='linear')

    metrics = get_accuracy(X_train, y_train, X_test, y_test, *classifiers)

    print('Rain Dataset')
    print(metrics)
    print()
    print()

def train_housing():

    dataset = HousingDataset()

    target = 'Beds'
    features = list(set(dataset.df.columns) - set([target]))

    y, X = dataset.get(features, target, ommit_nan=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifiers = train_all_classifiers(X_train, y_train, kernel='poly')

    metrics = get_accuracy(X_train, y_train, X_test, y_test, *classifiers)

    print('Housing Dataset')
    print(metrics)
    print()
    print()

def train_all():
    train_campus()
    train_heart()
    train_rain()
    train_housing()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay

def plot_decision_boundary(model, X, y):
    X_min, X_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(X_min, X_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


def confusion_matrix_table(model, X_test, y_test):
    """
    Generate a simple table representation of the confusion matrix.

    Parameters:
    - model: The trained classification model.
    - X_test: Test features.
    - y_test: True labels for the test set.

    Returns:
    - Confusion matrix table.
    """

    # Predictions
    y_pred = model.predict(X_test)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Convert to DataFrame for better visualization
    cm_df = pd.DataFrame(cm, columns=model.classes_, index=model.classes_)
    return cm_df


# plot_decision_boundary(model, X, y)
