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
            for n_steps in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
                trial = {
                    'Algorithm': 'RBO+',
                    'Parameters': {
                        'gamma': 0.05,
                        'step_size': 0.001,
                        'n_steps': n_steps,
                        'n_steps_scaling': 'linear'
                    },
                    'Classifier': classifier,
                    'Dataset': dataset,
                    'Fold': fold,
                    'Description': 'Preliminary (n_steps)'
                }

                databases.add_to_pending(trial)
