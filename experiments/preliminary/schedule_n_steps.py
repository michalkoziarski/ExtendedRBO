import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import databases
import datasets


for dataset in datasets.names('preliminary'):
    for fold in range(1, 11):
        for classifier in ['KNN', 'CART', 'SVM', 'NB']:
            for n_steps in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
                trial = {
                    'Algorithm': 'RBO+',
                    'Parameters': {
                        'gamma': 0.05,
                        'step_size': 0.001,
                        'n_steps': n_steps
                    },
                    'Classifier': classifier,
                    'Dataset': dataset,
                    'Fold': fold,
                    'Description': 'Preliminary (n_steps)'
                }

                databases.add_to_pending(trial)
