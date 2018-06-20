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
            for gamma_scaling in [None, 'linear', 'sqrt', 'log']:
                trial = {
                    'Algorithm': 'RBO+CV',
                    'Parameters': {
                        'gammas': [0.05, 0.5, 5.0],
                        'step_size': 0.001,
                        'n_steps': [1, 4, 16, 64, 256],
                        'n_steps_scaling': 'linear',
                        'gamma_scaling': gamma_scaling
                    },
                    'Classifier': classifier,
                    'Dataset': dataset,
                    'Fold': fold,
                    'Description': 'Preliminary (gamma_scaling)'
                }

                databases.add_to_pending(trial)
