import logging
import os
import sys

from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import databases
import datasets


logging.basicConfig(level=logging.INFO)

logging.info('Scheduling experiments...')


for dataset in tqdm(datasets.names('preliminary')):
    for fold in range(1, 11):
        for classifier in ['KNN', 'CART', 'SVM', 'NB']:
            for gamma in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
                trial = {
                    'Algorithm': 'RBO+CV',
                    'Parameters': {
                        'gamma': gamma,
                        'step_size': 0.001,
                        'n_steps': [1, 4, 16, 64, 256],
                        'n_steps_scaling': 'linear'
                    },
                    'Classifier': classifier,
                    'Dataset': dataset,
                    'Fold': fold,
                    'Description': 'Preliminary (gamma)'
                }

                databases.add_to_pending(trial)
