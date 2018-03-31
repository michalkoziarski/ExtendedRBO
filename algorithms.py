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
