import numpy as np


def distance(x, y, p=2):
    return np.sum(np.abs(x - y) ** p) ** (1 / p)


def rbf(d, gamma):
    if gamma == 0.0:
        return 0.0
    else:
        return np.exp(-(d / gamma) ** 2)


def mutual_class_potential(point, majority_points, minority_points, gamma):
    result = 0.0

    for majority_point in majority_points:
        result += rbf(distance(point, majority_point), gamma)

    for minority_point in minority_points:
        result -= rbf(distance(point, minority_point), gamma)

    return result


class RBO:
    def __init__(self, gamma=0.05, n_steps=500, step_size=0.001, n=None):
        self.gamma = gamma
        self.n_steps = n_steps
        self.step_size = step_size
        self.n = n

    def fit_sample(self, X, y):
        classes = np.unique(y)

        assert len(classes) == 2

        sizes = [sum(y == c) for c in classes]

        minority_class = classes[np.argmin(sizes)]
        majority_class = classes[np.argmax(sizes)]
        minority_points = X[y == minority_class].copy()
        majority_points = X[y == majority_class].copy()

        if self.n is None:
            n = len(majority_points) - len(minority_points)
        else:
            n = self.n

        appended = []

        while len(appended) < n:
            idx = np.random.choice(range(len(minority_points)))
            point = minority_points[idx].copy()
            potential = mutual_class_potential(point, majority_points, minority_points, self.gamma)

            for i in range(self.n_steps):
                translation = np.zeros(len(point))
                sign = np.random.choice([-1, 1])
                translation[np.random.choice(range(len(point)))] = sign * self.step_size
                translated_point = point + translation
                translated_potential = mutual_class_potential(translated_point, majority_points, minority_points, self.gamma)

                if np.abs(translated_potential) < np.abs(potential):
                    point = translated_point
                    potential = translated_potential

            appended.append(point)

        return np.concatenate([X, appended]), np.concatenate([y, minority_class * np.ones(len(appended))])


class FastRBO:
    def __init__(self, gamma=0.05, n_steps=500, step_size=0.001, cache_potential=True, n=None):
        self.gamma = gamma
        self.n_steps = n_steps
        self.step_size = step_size
        self.cache_potential = cache_potential
        self.n = n

    def fit_sample(self, X, y):
        classes = np.unique(y)

        assert len(classes) == 2

        sizes = [sum(y == c) for c in classes]

        minority_class = classes[np.argmin(sizes)]
        majority_class = classes[np.argmax(sizes)]
        minority_points = X[y == minority_class].copy()
        majority_points = X[y == majority_class].copy()

        if self.n is None:
            n = len(majority_points) - len(minority_points)
        else:
            n = self.n

        appended = []

        n_synthetic_points_per_minority_object = {i: 0 for i in range(len(minority_points))}

        for _ in range(n):
            idx = np.random.choice(range(len(minority_points)))
            n_synthetic_points_per_minority_object[idx] += 1

        for i in range(len(minority_points)):
            if self.cache_potential:
                cached_potentials = {}
            else:
                cached_potentials = None

            for _ in range(n_synthetic_points_per_minority_object[i]):
                point = minority_points[i].copy()
                translation = [0 for _ in range(len(point))]
                potential = self.fetch_or_compute_potential(point, translation, majority_points,
                                                            minority_points, cached_potentials)

                for _ in range(self.n_steps):
                    modified_translation = translation.copy()
                    sign = np.random.choice([-1, 1])
                    modified_translation[np.random.choice(range(len(point)))] += sign * self.step_size
                    modified_potential = self.fetch_or_compute_potential(point, modified_translation, majority_points,
                                                                         minority_points, cached_potentials)

                    if np.abs(modified_potential) < np.abs(potential):
                        translation = modified_translation
                        potential = modified_potential

                appended.append(point)

        return np.concatenate([X, appended]), np.concatenate([y, minority_class * np.ones(len(appended))])

    def fetch_or_compute_potential(self, point, translation, majority_points, minority_points, cached_potentials):
        if cached_potentials is None:
            return mutual_class_potential(point + translation, majority_points, minority_points, self.gamma)
        else:
            cached_potential = cached_potentials.get(tuple(translation))

            if cached_potential is None:
                potential = mutual_class_potential(point + translation, majority_points, minority_points, self.gamma)
                cached_potentials[tuple(translation)] = potential

                return potential
            else:
                return cached_potential
