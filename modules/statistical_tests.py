from scipy.stats import shapiro, pearsonr, kruskal, probplot, anderson, levene, bartlett, kstest, ttest_1samp, wilcoxon
from statsmodels.tsa.stattools import adfuller
import numpy as np

def shapiro_wilk_gaussian_test(array: np.ndarray):
    sw_stat, p_value = shapiro(array)

    alpha = 0.05

    print(f"Shapiro-Wilk Statistic: {sw_stat}")
    print(f"P-Value: {p_value}")

    if p_value < alpha:
        print(f"H0 rejected for an alpha of {alpha}, distribution can't be assumed drawn from a Normal")
    else:
        print(f"Fail to reject H0 for an alpha of {alpha}, possibly drawn from a Normal distribution")
    
    return sw_stat, p_value

def adf_test(array: np.ndarray):
    alpha = 0.05
    result = adfuller(array)

    adf_stat = result[0]
    p_value = result[1]

    print('ADF Statistic:', adf_stat)
    print('p-value:', p_value)

    if p_value < alpha:
        print(f"H0 rejected for an alpha of {alpha}. Series is possibly stationary")
    else:
        print(f"H0 not rejected for an alpha of {alpha}. Series is possibly non-stationary")

    return adf_stat, p_value

def anderson_darling_gaussian_test(data):
    """Critical values provided are for the following significance levels:

    normal/exponential
    15%, 10%, 5%, 2.5%, 1%

"""
    ad_stat, p_value, _ = anderson(data)

    alpha = 0.05

    print(f"Anderson-Darling Statistic: {ad_stat}")
    print(f"P-Value: {p_value[2]}")

    if p_value[2] < alpha:
        print(f"H0 rejected for an alpha of {alpha}, distribution can't be assumed drawn from a Normal")
    else:
        print(f"Fail to reject H0 for an alpha of {alpha}, possibly drawn from a Normal distribution")
    
    return ad_stat, p_value[2]

def ttest(buy_and_hold_sharpe, sharpes_distribution):
    t_stat, p_value = ttest_1samp(sharpes_distribution, buy_and_hold_sharpe, alternative="less")  # less -> dist mean is less than hypothesized value

    print(f"t-stat: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")

    alpha = 0.05
    if p_value < alpha:
        print("Reject H0: the evidence shows that the Sharpes' mean is less than B&H")
    else:
        print("Do not reject H0: there is not enough evidence to support the claim that B&H is superior to the model mean")
    
    return p_value

def kruskal_wallis_test(groups):
    alpha = 0.05

    h_statistic, p_value = kruskal(*groups)
    
    print(f"H-statistic: {h_statistic}")
    print(f"p-value: {p_value}")

    if p_value < alpha:
        print(f"There is a significant difference between at least two groups for an alpha of {alpha}")
    else:
        print(f"There is NOT a significant difference between groups for an alpha of {alpha}")

def wilcoxon_signed_rank(group, hypothesized_value):
    w_stat, p_value = wilcoxon(group - hypothesized_value, alternative='less')

    print(f"W statistic: {w_stat}")
    print(f"p-value: {p_value}")

    alpha = 0.05
    if p_value < alpha:
        print("Reject H0: the evidence shows that the Sharpes' mean is less than B&H")
    else:
        print("Do not reject H0: there is not enough evidence to support the claim that B&H is superior to the model mean")