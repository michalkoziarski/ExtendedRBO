import logging
import numpy as np

from itertools import product
from sklearn.model_selection import StratifiedKFold


def distance(x, y, p_norm=1):
    return np.sum(np.abs(x - y) ** p_norm) ** (1 / p_norm)


def rbf(d, gamma):
    if gamma == 0.0:
        return 0.0
    else:
        return np.exp(-(d / gamma) ** 2)


def mutual_class_potential(point, majority_points, minority_points, majority_gamma, minority_gamma=None):
    if minority_gamma is None:
        minority_gamma = majority_gamma

    result = 0.0

    for majority_point in majority_points:
        result += rbf(distance(point, majority_point), majority_gamma)

    for minority_point in minority_points:
        result -= rbf(distance(point, minority_point), minority_gamma)

    return result


def fetch_or_compute_potential(point, translation, majority_points, minority_points,
                               cached_potentials, majority_gamma, minority_gamma):
    if cached_potentials is None:
        return mutual_class_potential(point + translation, majority_points, minority_points,
                                      majority_gamma, minority_gamma)
    else:
        cached_potential = cached_potentials.get(tuple(translation))

        if cached_potential is None:
            potential = mutual_class_potential(point + translation, majority_points, minority_points,
                                               majority_gamma, minority_gamma)
            cached_potentials[tuple(translation)] = potential

            return potential
        else:
            return cached_potential


def generate_possible_directions(n_dimensions, excluded_direction=None):
    possible_directions = []

    for dimension in range(n_dimensions):
        for sign in [-1, 1]:
            if excluded_direction is None or (excluded_direction[0] != dimension or excluded_direction[1] != sign):
                possible_directions.append((dimension, sign))

    np.random.shuffle(possible_directions)

    return possible_directions


class ExtendedRBO:
    def __init__(self, gamma=0.05, step_size=0.001, n_steps=500, gamma_scaling='none', n_steps_scaling='none',
                 approximate_potential=False, n_nearest_neighbors=25, borderline=False, m_nearest_neighbors=0.5,
                 ignore_outliers=False, k_nearest_neighbors=5, generate_in_between=False, cache_potential=True,
                 n=None):
        assert gamma_scaling in ['none', 'linear', 'sqrt', 'log']
        assert n_steps_scaling in ['none', 'linear']
        assert not (borderline and ignore_outliers)

        self.gamma = gamma
        self.step_size = step_size
        self.n_steps = n_steps
        self.gamma_scaling = gamma_scaling
        self.n_steps_scaling = n_steps_scaling
        self.approximate_potential = approximate_potential
        self.n_nearest_neighbors = n_nearest_neighbors
        self.borderline = borderline
        self.m_nearest_neighbors = m_nearest_neighbors
        self.ignore_outliers = ignore_outliers
        self.k_nearest_neighbors = k_nearest_neighbors
        self.generate_in_between = generate_in_between
        self.cache_potential = cache_potential
        self.n = n

    def fit_sample(self, X, y):
        classes = np.unique(y)

        assert len(classes) == 2

        sizes = [sum(y == c) for c in classes]

        minority_class = classes[np.argmin(sizes)]
        majority_class = classes[np.argmax(sizes)]
        minority_points = X[y == minority_class]
        majority_points = X[y == majority_class]

        imbalance_ratio = len(majority_points) / len(minority_points)

        majority_gamma = self.gamma

        if self.gamma_scaling == 'none':
            minority_gamma = self.gamma
        elif self.gamma_scaling == 'linear':
            minority_gamma = self.gamma * imbalance_ratio
        elif self.gamma_scaling == 'sqrt':
            minority_gamma = self.gamma * np.sqrt(imbalance_ratio)
        elif self.gamma_scaling == 'log':
            minority_gamma = self.gamma * np.log2(1 + imbalance_ratio)
        else:
            raise NotImplementedError

        if self.n_steps_scaling == 'none':
            n_steps = self.n_steps
        elif self.n_steps_scaling == 'linear':
            n_steps = self.n_steps * X.shape[1]
        else:
            raise NotImplementedError

        if self.n is None:
            n = len(majority_points) - len(minority_points)
        else:
            n = self.n

        appended = []

        if self.borderline:
            sorted_neighbors_indices = []
            considered_minority_points_indices = []

            for i in range(len(minority_points)):
                distance_vector = [distance(minority_points[i], x) for x in X]
                distance_vector[i] = -np.inf
                indices = np.argsort(distance_vector)
                sorted_neighbors_indices.append(indices)

            if type(self.m_nearest_neighbors) is int:
                considered_m_values = [self.m_nearest_neighbors]
            elif type(self.m_nearest_neighbors) is float:
                considered_m_values = list(range(len(X)))
            else:
                raise NotImplementedError

            n_borderline = []

            for m in considered_m_values:
                n_borderline.append(0)

                for i in range(len(minority_points)):
                    n_minority_neighbors = np.sum(
                        y[sorted_neighbors_indices[i][1:(m + 1)]] == minority_class
                    )

                    if m / 2 <= n_minority_neighbors < m:
                        n_borderline[-1] += 1

            borderline_fractions = [n_b / len(minority_points) for n_b in n_borderline]
            selected_idx = np.argmin([np.abs(self.m_nearest_neighbors - fraction) for fraction in borderline_fractions])
            m = considered_m_values[selected_idx]

            for i in range(len(minority_points)):
                n_minority_neighbors = np.sum(
                    y[sorted_neighbors_indices[i][1:(m + 1)]] == minority_class
                )

                if m / 2 <= n_minority_neighbors < m:
                    considered_minority_points_indices.append(i)

            if len(considered_minority_points_indices) == 0:
                logging.warning('Failed to find any borderline instances, falling back to basic mode.')

                considered_minority_points_indices = range(len(minority_points))
        elif self.ignore_outliers:
            sorted_neighbors_indices = []
            considered_minority_points_indices = []

            for i in range(len(minority_points)):
                distance_vector = [distance(minority_points[i], x) for x in X]
                distance_vector[i] = -np.inf
                indices = np.argsort(distance_vector)
                sorted_neighbors_indices.append(indices)
                n_minority_neighbors = np.sum(y[indices[1:(self.k_nearest_neighbors + 1)]] == minority_class)

                if n_minority_neighbors > 0:
                    considered_minority_points_indices.append(i)

            if len(considered_minority_points_indices) == 0:
                logging.warning('Failed to find any non-outlier instances, falling back to basic mode.')

                considered_minority_points_indices = range(len(minority_points))
        else:
            sorted_neighbors_indices = None
            considered_minority_points_indices = range(len(minority_points))

        n_synthetic_points_per_minority_object = {i: 0 for i in considered_minority_points_indices}

        for _ in range(n):
            idx = np.random.choice(considered_minority_points_indices)
            n_synthetic_points_per_minority_object[idx] += 1

        for i in considered_minority_points_indices:
            if n_synthetic_points_per_minority_object[i] == 0:
                continue

            point = minority_points[i]

            if self.cache_potential:
                cached_potentials = {}
            else:
                cached_potentials = None

            if self.approximate_potential:
                if sorted_neighbors_indices is None:
                    distance_vector = [distance(point, x) for x in X]
                    distance_vector[i] = -np.inf
                    indices = np.argsort(distance_vector)[:(self.n_nearest_neighbors + 1)]
                else:
                    indices = sorted_neighbors_indices[i][:(self.n_nearest_neighbors + 1)]

                closest_points = X[indices]
                closest_labels = y[indices]
                closest_minority_points = closest_points[closest_labels == minority_class]
                closest_majority_points = closest_points[closest_labels == majority_class]
            else:
                closest_minority_points = minority_points
                closest_majority_points = majority_points

            for _ in range(n_synthetic_points_per_minority_object[i]):
                translation = [0 for _ in range(len(point))]
                translation_history = [translation]
                potential = fetch_or_compute_potential(point, translation, closest_majority_points,
                                                       closest_minority_points, cached_potentials,
                                                       majority_gamma, minority_gamma)
                possible_directions = generate_possible_directions(len(point))

                for _ in range(n_steps):
                    if len(possible_directions) == 0:
                        break

                    dimension, sign = possible_directions.pop()
                    modified_translation = translation.copy()
                    modified_translation[dimension] += sign * self.step_size
                    modified_potential = fetch_or_compute_potential(point, modified_translation,
                                                                    closest_majority_points, closest_minority_points,
                                                                    cached_potentials, majority_gamma, minority_gamma)

                    if np.abs(modified_potential) < np.abs(potential):
                        translation = modified_translation
                        translation_history.append(translation)
                        potential = modified_potential
                        possible_directions = generate_possible_directions(len(point), (dimension, -sign))

                if self.generate_in_between:
                    translation = translation_history[np.random.randint(len(translation_history))]

                appended.append(point + translation)

        return np.concatenate([X, appended]), np.concatenate([y, minority_class * np.ones(len(appended))])


class ExtendedRBOCV:
    def __init__(self, classifier, measure, n_splits=3, n_steps=(500, ), gammas=(0.05, ), **kwargs):
        self.classifier = classifier
        self.measure = measure
        self.n_splits = n_splits
        self.n_steps = n_steps
        self.gammas = gammas
        self.kwargs = kwargs

        self.skf = StratifiedKFold(n_splits=n_splits)

    def fit_sample(self, X, y):
        self.skf.get_n_splits(X, y)

        best_score = -np.inf
        best_n_steps = None
        best_gamma = None

        for n_steps in self.n_steps:
            for gamma in self.gammas:
                scores = []

                for train_idx, test_idx in self.skf.split(X, y):
                    X_train, y_train = ExtendedRBO(n_steps=n_steps, gamma=gamma, **self.kwargs).\
                        fit_sample(X[train_idx], y[train_idx])

                    classifier = self.classifier.fit(X_train, y_train)
                    predictions = classifier.predict(X[test_idx])
                    scores.append(self.measure(y[test_idx], predictions))

                score = np.mean(scores)

                if score > best_score:
                    best_score = score
                    best_n_steps = n_steps
                    best_gamma = gamma

        return ExtendedRBO(n_steps=best_n_steps, gamma=best_gamma, **self.kwargs).fit_sample(X, y)


class ResamplingCV:
    def __init__(self, algorithm, classifier, measure, n_splits=3, **kwargs):
        self.algorithm = algorithm
        self.classifier = classifier
        self.measure = measure
        self.n_splits = n_splits
        self.kwargs = kwargs

        self.skf = StratifiedKFold(n_splits=n_splits)

    def fit_sample(self, X, y):
        self.skf.get_n_splits(X, y)

        best_score = -np.inf
        best_parameters = None

        for parameters in (dict(zip(self.kwargs, x)) for x in product(*self.kwargs.values())):
            scores = []

            for train_idx, test_idx in self.skf.split(X, y):
                X_train, y_train = self.algorithm(**parameters).fit_sample(X[train_idx], y[train_idx])

                classifier = self.classifier.fit(X_train, y_train)
                predictions = classifier.predict(X[test_idx])
                scores.append(self.measure(y[test_idx], predictions))

            score = np.mean(scores)

            if score > best_score:
                best_score = score
                best_parameters = parameters

        logging.info('Selected parameters: %s.' % best_parameters)

        return self.algorithm(**best_parameters).fit_sample(X, y)
