from collections import OrderedDict
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from pprint import pprint

from sklearn.model_selection import train_test_split

from utils.data import HeartDataset, HousingDataset, RainDataset, CampusDataset

from time import perf_counter as time
import polars as pl
import numpy as np
import os

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support

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

    start_time = time()
    gbc = train_gradient_boosting_classifier(X, y, n_estimators=n_estimators)
    training_times['GradientBoosting'] = time() - start_time

    start_time = time()
    lrc = train_logistic_regression(X, y)
    training_times['LogisticRegression'] = time() - start_time

    start_time = time()
    rfc = train_random_forest_classifier(X, y, n_estimators=n_estimators)
    training_times['RandomForest'] = time() - start_time

    start_time = time()
    svc = train_support_vector_classifier(X, y, kernel)
    training_times['SupportVector'] = time() - start_time

    start_time = time()
    nnc = train_neural_network_classifier(X, y)
    training_times['NeuralNetwork'] = time() - start_time

    return gbc, lrc, rfc, svc, nnc, training_times

def get_accuracy(X_train, y_train, X_test, y_test, gbc, lrc, rfc, svc, nnc, training_times):

    metrics = OrderedDict(
        accuracy = ['Test CCR', 'Train CCR', 'Test Precision', 'Test Recall', 'F1 score', 'Training Time'],
        grad_boost = [
            gbc.score(X_test, y_test),
            gbc.score(X_train, y_train),
            *precision_recall_fscore_support(y_test, gbc.predict(X_test), average='weighted')[:-1],
            training_times['GradientBoosting']
        ],
        log_reg = [
            lrc.score(X_test, y_test),
            lrc.score(X_train, y_train),
            *precision_recall_fscore_support(y_test, lrc.predict(X_test), average='weighted')[:-1],
            training_times['LogisticRegression']
        ],
        rand_forest = [
            rfc.score(X_test, y_test),
            rfc.score(X_train, y_train),
            *precision_recall_fscore_support(y_test, rfc.predict(X_test), average='weighted')[:-1],
            training_times['RandomForest']
        ],
        svc = [
            svc.score(X_test, y_test),
            svc.score(X_train, y_train),
            *precision_recall_fscore_support(y_test, svc.predict(X_test), average='weighted')[:-1],
            training_times['SupportVector']
        ],
        neural_net = [
            nnc.score(X_test, y_test),
            nnc.score(X_train, y_train),
            *precision_recall_fscore_support(y_test, nnc.predict(X_test), average='weighted')[:-1],
            training_times['NeuralNetwork']
        ]
    )

    return pl.DataFrame(metrics)

def plot_confusion_matrix(X_test, y_test, basedir, labels, *models):
    """
    Generate a Polars DataFrame representation of the confusion matrix.

    Parameters:
    - model: The trained classification model.
    - X_test: Test features.
    - y_test: True labels for the test set.

    Returns:
    - Confusion matrix table as a Polars DataFrame.
    """

    plt.figure()

    os.makedirs(f'{basedir}/confusion', exist_ok=True)

    for model, name in zip(models, ['GradientBoosting', 'LogisticRegression', 'RandomForest', 'SupportVector', 'NeuralNetwork']):

        # Predictions
        y_pred = model.predict(X_test)

        y_pred_final = [labels[i] for i in y_pred]
        y_test_final = [labels[i] for i in y_test]

        # Calculate confusion matrix
        cm = confusion_matrix(y_test_final, y_pred_final, labels=labels)

        # Convert to Polars DataFrame for better visualization
        cm_df = ConfusionMatrixDisplay(cm, display_labels=labels)

        cm_df.plot()

        plt.savefig(f'{basedir}/confusion/{name}.png')

def plot_class_balance(Y, basedir, xlabel, xlabel_ticks):
    """
    Plot the class balance of the given dataset.

    Parameters:
    - Y: The target variable.
    """

    plt.figure()

    y_in = Y.to_list()

    counts, bins, patches = plt.hist(y_in, bins=np.arange(min(y_in)-0.5, max(y_in)+1, 1), rwidth=0.5)

    # Add count annotations above each bar
    for count, bin, patch in zip(counts, bins, patches):
        if count > 0:
            plt.text(bin+0.5, patch.get_height(), str(int(count)), ha='center', va='bottom')

    plt.ylabel('Count')
    plt.xlabel(xlabel)
    plt.xticks(np.arange(len(Y.unique())), labels=xlabel_ticks)

    plt.savefig(f'{basedir}/class_balance.png')

def train_campus():

    os.makedirs('results/campus', exist_ok=True)

    dataset = CampusDataset()

    target = 'degree_t'
    features = list(set(dataset.df.columns) - set([target, 'salary']))

    y, X = dataset.get(features, target, ommit_nan=True)

    xlabel_ticks = [dataset.categorical[target][i] for i in sorted(y.unique().to_list())]
    plot_class_balance(y, 'results/campus', 'Degree placement', xlabel_ticks)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifiers = train_all_classifiers(X_train, y_train, kernel='linear')

    metrics = get_accuracy(X_train, y_train, X_test, y_test, *classifiers)

    plot_confusion_matrix(X_test, y_test, 'results/campus', xlabel_ticks, *classifiers)

    metrics.write_csv('results/campus/metrics.csv')

    print('Campus Placement Dataset')
    print(metrics)
    print()
    print()

def train_heart():

    os.makedirs('results/heart', exist_ok=True)
    
    dataset = HeartDataset()

    target = 'output'
    features = list(set(dataset.df.columns) - set([target]))

    y, X = dataset.get(features, target, ommit_nan=True)

    xlabel_ticks = [dataset.categorical[target][i] for i in sorted(y.unique().to_list())]
    plot_class_balance(y, 'results/heart', 'Heart Attack Risk', xlabel_ticks)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifiers = train_all_classifiers(X_train, y_train, kernel='poly')

    metrics = get_accuracy(X_train, y_train, X_test, y_test, *classifiers)

    plot_confusion_matrix(X_test, y_test, 'results/heart', xlabel_ticks, *classifiers)

    metrics.write_csv('results/heart/metrics.csv')

    print('Heart Disease Dataset')
    print(metrics)
    print()
    print()

def train_rain():

    os.makedirs('results/rain', exist_ok=True)

    dataset = RainDataset()

    target = 'RainTomorrow'
    features = list(set(dataset.df.columns) - set([target]))

    y, X = dataset.get(features, target, ommit_nan=True)

    xlabel_ticks = [dataset.categorical[target][i] for i in sorted(y.unique().to_list())]
    plot_class_balance(y, 'results/rain', 'Rain Tomorrow', xlabel_ticks)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifiers = train_all_classifiers(X_train, y_train, n_estimators=500, kernel='linear')

    metrics = get_accuracy(X_train, y_train, X_test, y_test, *classifiers)

    plot_confusion_matrix(X_test, y_test, 'results/rain', xlabel_ticks, *classifiers)

    metrics.write_csv('results/rain/metrics.csv')

    print('Rain Dataset')
    print(metrics)
    print()
    print()

def train_housing():

    os.makedirs('results/housing', exist_ok=True)

    dataset = HousingDataset()

    target = 'Beds'
    features = list(set(dataset.df.columns) - set([target]))

    y, X = dataset.get(features, target, ommit_nan=True)

    xlabel_ticks = [dataset.categorical[target][i] for i in sorted(y.unique().to_list())]
    plot_class_balance(y, 'results/housing', 'Number of Beds', xlabel_ticks)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifiers = train_all_classifiers(X_train, y_train, kernel='rbf')

    metrics = get_accuracy(X_train, y_train, X_test, y_test, *classifiers)

    plot_confusion_matrix(X_test, y_test, 'results/housing', xlabel_ticks, *classifiers)

    metrics.write_csv('results/housing/metrics.csv')

    print('Housing Dataset')
    print(metrics)
    print()
    print()

def train_all():
    train_campus()
    train_heart()
    train_rain()
    train_housing()
