from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from utils.data import HeartDataset, HousingDataset, RainDataset, CampusDataset

def train_gradient_boosting_classifier(X, y):
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
    gbc = GradientBoostingClassifier(n_estimators=1_000, learning_rate=1e0, max_depth=1, random_state=42)

    # Train the classifier
    gbc.fit(X, y)

    return gbc

def train_campus():

    dataset = CampusDataset()
    
    target = 'degree_t'
    features = list(set(dataset.df.columns) - set([target, 'salary']))

    y, X = dataset.get(features, target, ommit_nan=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    gbc = train_gradient_boosting_classifier(X_train, y_train)

    metrics = {}

    metrics['grad_boost'] = {
        'test_accuracy': gbc.score(X_test, y_test),
        'train_accuracy': gbc.score(X_train, y_train)
    }

    print('Campus Placement Dataset')
    print(metrics)

def train_heart():
    
    dataset = HeartDataset()

    target = 'output'
    features = list(set(dataset.df.columns) - set([target]))

    y, X = dataset.get(features, target, ommit_nan=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    gbc = train_gradient_boosting_classifier(X_train, y_train)

    metrics = {}

    metrics['grad_boost'] = {
        'test_accuracy': gbc.score(X_test, y_test),
        'train_accuracy': gbc.score(X_train, y_train)
    }

    print('Heart Disease Dataset')
    print(metrics)

def train_rain():

    dataset = RainDataset()

    target = 'RainTomorrow'
    features = list(set(dataset.df.columns) - set([target]))

    y, X = dataset.get(features, target, ommit_nan=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    gbc = train_gradient_boosting_classifier(X_train, y_train)

    metrics = {}

    metrics['grad_boost'] = {
        'test_accuracy': gbc.score(X_test, y_test),
        'train_accuracy': gbc.score(X_train, y_train)
    }

    print('Rain Dataset')
    print(metrics)

def train_housing():

    dataset = HousingDataset()

    target = 'Beds'
    features = list(set(dataset.df.columns) - set([target]))

    y, X = dataset.get(features, target, ommit_nan=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    gbc = train_gradient_boosting_classifier(X_train, y_train)

    metrics = {}

    metrics['grad_boost'] = {
        'test_accuracy': gbc.score(X_test, y_test),
        'train_accuracy': gbc.score(X_train, y_train)
    }

    print('Housing Dataset')
    print(metrics)

def train_all():
    train_campus()
    train_heart()
    train_rain()
    train_housing()
