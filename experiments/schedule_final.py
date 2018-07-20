import logging
import os
import sys

from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import databases
import datasets


logging.basicConfig(level=logging.INFO)

logging.info('Scheduling experiments...')


for dataset in tqdm(datasets.names('final')):
    for fold in range(1, 11):
        for classifier in ['KNN', 'CART', 'SVM', 'NB']:
            trial = {
                'Algorithm': 'RBO+CV',
                'Parameters': {
                    'gammas': [0.05, 0.5, 5.0],
                    'step_size': 0.001,
                    'n_steps': [1, 4, 16, 64, 256],
                    'approximate_potential': True,
                    'n_nearest_neighbors': 25,
                    'generate_in_between': True,
                    'n_steps_scaling': 'linear'
                },
                'Classifier': classifier,
                'Dataset': dataset,
                'Fold': fold,
                'Description': 'Final'
            }

            databases.add_to_pending(trial)

            for algorithm in ['ROS', 'RUS', 'SMOTE', 'SMOTE+ENN', 'SMOTE+TL', 'Bord', 'ADASYN', 'NCL']:
                trial = {
                    'Algorithm': algorithm,
                    'Parameters': None,
                    'Classifier': classifier,
                    'Dataset': dataset,
                    'Fold': fold,
                    'Description': 'Final'
                }

                databases.add_to_pending(trial)
