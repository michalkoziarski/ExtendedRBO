import databases
import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from collections import OrderedDict
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr


def load_preliminary_dict(classifier, metric, parameter, values):
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


def load_preliminary_df(classifier, parameter, values, metrics=('F-measure', 'AUC', 'G-mean')):
    trials = pd.DataFrame(databases._select({
        'Algorithm': 'RBO+',
        'Description': 'Preliminary (%s)' % parameter,
        'Classifier': classifier
    }, fetch='all', database_path=databases.FINISHED_PATH))

    rows = []

    for dataset in datasets.names('preliminary'):
        dataset_selection = trials[trials['Dataset'] == dataset]

        for value in values:
            selection = dataset_selection[dataset_selection['Parameters'].str.contains("'%s': %s," % (parameter, value))]
            scores = list(selection['Scores'].map(lambda x: eval(x)))

            assert len(scores) == 10

            for metric in metrics:
                score = np.mean([score[metric] for score in scores])
                rows.append([dataset, metric, value, score])

    return pd.DataFrame(rows, columns=['DS', 'metric', 'value', 'score'])


def test_friedman_shaffer(dictionary):
    df = pd.DataFrame(dictionary)

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


def plot_preliminary(classifier, parameter, values, metrics=('F-measure', 'AUC', 'G-mean'), outname=None, xlabel=None):
    if xlabel is None:
        xlabel = parameter.replace('_', ' ')

    df = load_preliminary_df(classifier, parameter, values, metrics)
    df[xlabel] = df['value'].map(lambda x: values.index(x))

    grid = sns.FacetGrid(df, col='DS', hue='metric', col_wrap=5)
    grid.map(plt.plot, xlabel, 'score')
    grid.set(xticks=list(range(len(values))), xticklabels=values, ylim=(0.0, 1.0))
    grid.fig.legend(loc='lower center', ncol=3, labels=metrics)
    grid.fig.subplots_adjust(bottom=0.075)

    if outname is None:
        plt.show()
    else:
        plt.savefig(outname, bbox_inches='tight')
