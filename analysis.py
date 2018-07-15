import databases
import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from collections import OrderedDict
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr


def _get_trials(experiment, classifier, algorithm='RBO+'):
    trials = pd.DataFrame(databases._select({
        'Algorithm': algorithm,
        'Description': experiment,
        'Classifier': classifier
    }, fetch='all', database_path=databases.FINISHED_PATH))

    return trials


def _select_scores(df, parameter, value):
    selection = df[df['Parameters'].str.contains("'%s': %s[,|}]" % (parameter, value))]
    scores = list(selection['Scores'].map(lambda x: eval(x)))

    assert len(scores) == 10

    return scores


def load_preliminary_dict(classifier, metric, parameter, values, algorithm='RBO+'):
    trials = _get_trials('Preliminary (%s)' % parameter, classifier, algorithm)

    measurements = OrderedDict()

    for value in values:
        measurements[value] = []

    for dataset in datasets.names('preliminary'):
        dataset_selection = trials[trials['Dataset'] == dataset]

        for value in values:
            scores = _select_scores(dataset_selection, parameter, value)
            score = np.mean([score[metric] for score in scores])
            measurements[value].append(score)

    return measurements


def load_preliminary_df(classifier, parameter, values, metrics=('F-measure', 'AUC', 'G-mean'), algorithm='RBO+'):
    trials = _get_trials('Preliminary (%s)' % parameter, classifier, algorithm)

    rows = []

    for dataset in datasets.names('preliminary'):
        dataset_selection = trials[trials['Dataset'] == dataset]

        for value in values:
            scores = _select_scores(dataset_selection, parameter, value)

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


def plot_preliminary(classifier, parameter, values, outname=None, kind='miniplots',
                     metrics=('F-measure', 'AUC', 'G-mean'), xlabel=None, ylim=(0.0, 1.0), algorithm=None):
    assert kind in ['miniplots', 'pointplots', 'boxplots', 'barplots']

    if xlabel is None:
        xlabel = parameter.replace('_', ' ')

    if algorithm is None:
        if parameter == 'n_steps':
            algorithm = 'RBO+'
        else:
            algorithm = 'RBO+CV'

    df = load_preliminary_df(classifier, parameter, values, metrics, algorithm)
    df[xlabel] = df['value'].map(lambda x: values.index(x))

    if kind == 'miniplots':
        grid = sns.FacetGrid(df, col='DS', hue='metric', col_wrap=5)
        grid.map(plt.plot, xlabel, 'score')
        grid.fig.legend(loc='lower center', ncol=3, labels=metrics)
        grid.fig.subplots_adjust(bottom=0.075)
    elif kind == 'pointplots':
        grid = sns.factorplot(x=xlabel, y='score', hue='metric', data=df, kind='point')
    elif kind == 'boxplots':
        grid = sns.boxplot(x=xlabel, y='score', hue='metric', data=df)
    elif kind == 'barplots':
        grid = sns.factorplot(x=xlabel, y='score', hue='metric', data=df, kind='bar')
    else:
        raise ValueError

    grid.set(xticks=list(range(len(values))), xticklabels=values, ylim=ylim)

    if outname is None:
        plt.show()
    else:
        plt.savefig(outname, bbox_inches='tight')
