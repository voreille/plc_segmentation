from re import I
import numpy as np
import pandas as pd
from scipy.stats import t, chi2, binom
from scipy.special import comb
from sklearn.metrics import confusion_matrix


def corrected_std(differences, n_train, n_test):
    kr = len(differences)
    corrected_var = np.var(differences, ddof=1) * (1 / kr + n_test / n_train)
    corrected_std = np.sqrt(corrected_var)
    return corrected_std


def compute_corrected_ttest(differences, df, n_train, n_test):
    mean = np.mean(differences)
    std = corrected_std(differences, n_train, n_test)
    t_stat = mean / std
    p_val = t.sf(np.abs(t_stat), df)  # right-tailed t-test
    # p_val = t.sf(t_stat, df)
    return t_stat, p_val


def compare_score(s1, s2, n_train, n_test, rope_interval=[-0.01, 0.01]):
    n = s1.shape[0]

    t_stat, p_val = compute_corrected_ttest(s1 - s2, n - 1, n_train, n_test)

    t_post = t(n - 1,
               loc=np.mean(s1 - s2),
               scale=corrected_std(s1 - s2, n_train, n_test))

    ri = t_post.cdf(rope_interval[1]) - t_post.cdf(rope_interval[0])
    return {
        "t_stat": t_stat,
        "p_value": p_val,
        "proba M1 > M2": 1 - t_post.cdf(rope_interval[1]),
        "proba M1 == M2": ri,
        "proba M1 < M2": t_post.cdf(rope_interval[0]),
    }


def corrected_ci(s, n_train, n_test, alpha=0.95):
    n = s.shape[0]
    mean = np.mean(s)
    std = corrected_std(s, n_train, n_test)
    return t.interval(alpha, n - 1, loc=mean, scale=std)


# def compute_mcnemar_test(x1, x2):
#     """
#    Compute the McNemar test for two binary classification problems.
#    seems to be wrong though, code copied from 
#    https://aaronschlegel.me/mcnemars-test-paired-data-python.html
#     """
#     cm = confusion_matrix(x1, x2)
#     i = cm[0, 1]
#     n = cm[1, 0] + cm[0, 1]
#     i_n = np.arange(i + 1, n + 1)
#     p_value_exact = 1 - np.sum(comb(n, i_n) * 0.5**i_n * (1 - 0.5)**(n - i_n))
#     p_value_exact *= 2
#     mid_p_value = p_value_exact - binom.pmf(cm[0, 1], n, 0.5)

#     x2_statistic = (np.absolute(cm[0, 1] - cm[1, 0]) - 1)**2 / (cm[0, 1] +
#                                                                 cm[1, 0])
#     p_value_corrected = chi2.sf(x2_statistic, 1)

#     x2_statistic = (cm[0, 1] - cm[1, 0])**2 / (cm[0, 1] + cm[1, 0])
#     p_value = chi2.sf(x2_statistic, 1)

#     return {
#         "p_value": p_value,
#         "p_value_corrected": p_value_corrected,
#         "p_value_exact": p_value_exact,
#         "p_value_mid": mid_p_value,
#     }