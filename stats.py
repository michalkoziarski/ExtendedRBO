import numpy as np

from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr


def friedman_shaffer(df):
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
