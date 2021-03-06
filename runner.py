import json
import logging
import multiprocessing as mp
import numpy as np
import time

from algorithms import ExtendedRBO, ExtendedRBOCV
from databases import pull_pending, submit_result
from datasets import load
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from imblearn.under_sampling import NeighbourhoodCleaningRule, RandomUnderSampler


logging.basicConfig(level=logging.INFO)

N_PROCESSES = 24


def run():
    while True:
        trial = pull_pending()

        if trial is None:
            break

        logging.info('Processing trial: %s...' % trial)

        dataset = load(trial['Dataset'])
        fold = int(trial['Fold']) - 1

        (X_train, y_train), (X_test, y_test) = dataset[fold][0], dataset[fold][1]

        labels = np.unique(y_test)
        counts = [len(y_test[y_test == label]) for label in labels]
        minority_class = labels[np.argmin(counts)]

        k_neighbors = np.min([len(y_train[y_train == minority_class]) - 1, 5])

        classifiers = {
            'NB': GaussianNB(),
            'KNN': KNeighborsClassifier(),
            'SVM': SVC(),
            'CART': DecisionTreeClassifier()
        }

        algorithms = {
            'ROS': RandomOverSampler(),
            'RUS': RandomUnderSampler(),
            'SMOTE': SMOTE(k=k_neighbors),
            'SMOTE+ENN': SMOTEENN(k=k_neighbors),
            'SMOTE+TL': SMOTETomek(k=k_neighbors),
            'Bord': SMOTE(k=k_neighbors, kind='borderline1'),
            'ADASYN': ADASYN(k=k_neighbors),
            'NCL': NeighbourhoodCleaningRule()
        }

        clf = classifiers.get(trial['Classifier'])

        if trial['Algorithm'] in ['RBO+', 'RBO+CV']:
            if trial.get('Parameters') is None:
                params = {}
            else:
                params = json.loads(
                    trial['Parameters'].replace('\'', '"').replace('False', 'false').replace('True', 'true')
                )

            if trial['Algorithm'] == 'RBO+':
                algorithm = ExtendedRBO(**params)
            elif trial['Algorithm'] == 'RBO+CV':
                algorithm = ExtendedRBOCV(clf, metrics.roc_auc_score, **params)
            else:
                raise NotImplementedError
        elif (trial['Algorithm'] is None) or (trial['Algorithm'] == 'None'):
            algorithm = None
        else:
            algorithm = algorithms.get(trial['Algorithm'])

            if algorithm is None:
                raise NotImplementedError

        start_time = time.process_time()

        if algorithm is not None:
            X_train, y_train = algorithm.fit_sample(X_train, y_train)

        end_time = time.process_time()

        clf = clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        g_mean = 1.0

        for label in np.unique(y_test):
            idx = (y_test == label)
            g_mean *= metrics.accuracy_score(y_test[idx], predictions[idx])

        g_mean = np.sqrt(g_mean)

        scores = {
            'Accuracy': metrics.accuracy_score(y_test, predictions),
            'Precision': metrics.precision_score(y_test, predictions, pos_label=minority_class),
            'Recall': metrics.recall_score(y_test, predictions, pos_label=minority_class),
            'F-measure': metrics.f1_score(y_test, predictions, pos_label=minority_class),
            'AUC': metrics.roc_auc_score(y_test, predictions),
            'G-mean': g_mean,
            'Time': end_time - start_time
        }

        submit_result(trial, scores)


if __name__ == '__main__':
    for _ in range(N_PROCESSES):
        mp.Process(target=run).start()
