import databases
import datasets
import numpy as np
import pandas as pd

from collections import OrderedDict
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr


def load_preliminary_measurements(classifier, metric, parameter, values):
    trials = pd.DataFrame(databases._select({
        'Algorithm': 'RBO+',
        'Description': 'Preliminary (%s)' % parameter,
        'Classifier': classifier
    }, fetch='all', database_path=databases.FINISHED_PATH))

    measurements = OrderedDict()

    for value in values:
        measurements[value] = []

    for dataset in datasets.names('preliminary'):
        dataset_selection = trials[trials['Dataset'] == dataset]

        for value in values:
            selection = dataset_selection[dataset_selection['Parameters'].str.contains("'%s': %s," % (parameter, value))]
            scores = list(selection['Scores'].map(lambda x: eval(x)))

            assert len(scores) == 10

            score = np.mean([score[metric] for score in scores])
            measurements[value].append(score)

    return measurements


def test_friedman_shaffer(df):
    pandas2ri.activate()

    importr("scmamp")

    rFriedmanTest = r['friedmanTest']
    rPostHocTest = r['postHocTest']

    initial_results = rFriedmanTest(df)
    posthoc_results = rPostHocTest(df, test="friedman", correct="shaffer", use_rank=True)

    p_value = initial_results[2][0]
    ranks = np.array(posthoc_results[0])[0]
    corrected_p_values = np.array(posthoc_results[2])

    return ranks, p_value, corrected_p_values
