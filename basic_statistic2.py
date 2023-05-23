import numpy as np
import matplotlib.pyplot as plt
import math
import sklearn
from sklearn.linear_model import LinearRegression
import scipy
import statistics
import pandas as pd
import random
'''
Basic statistical terms every data analyst must 
know
1.Term name
2.My function
3.Python functions from modules above 
Below lists represent variables
'''
xies = [1,2,3,4,5,6,7,8,9]
yies = [2,4,6,8,10,12,14,16,18]

#Arithmetic average:
def average(list):
    arithmetic_average = sum(list)/len(list)
    return arithmetic_average
#print(average(xies))


#Modes
def modes(list):
    count_dict = {}
    for num in list:
        if num in count_dict:
            count_dict[num] += 1
        else:
            count_dict[num] = 1
    max_count = max(count_dict.values())
    modes = [num for num, count in count_dict.items() if count == max_count]
    return modes
#print(modes(xies))


#Median:
def median(list):
    list.sort()
    centre = len(list)/2
    if len(list) % 2 == 0:
        p1 = list[int(centre)]
        p2 = list[int(centre-1)]
        pp = (p1+p2)/2
        return pp
    else:
        return list[int(centre)]
#print(median(xies))

    
#Median absolute deviation from the median
def mad(list):
    l = []
    for x in list:
        l.append(abs(x - median(list))) 
    return median(l)
#print(mad(xies))


#Variance
def variance(list):
    result = []
    for i in list:
        result.append((i - (average(list)))**2)
    return (sum(result))/(len(list)-1)
#print(variance(xies))
#print(np.var(xies))
#print(statistics.variance(xies))
#pandas.DataFrame.var()
#scipy.stats.variation()
#math.fsum(), math.pow()


#Standard deviation
def deviation(list):
    devi = np.sqrt(variance(list))
    return devi
#print(deviation(xies))


#Standard error
def se(list):
    return deviation(list)/np.sqrt(len(list))
#print("Standard Error: ",se(xies))


#Skewness
def skewness(list):
    skew = 3*(average(list) - median(list))/deviation(list)
    return skew
#print(skewness(xies))
#scipy.stats.skew()
#print(np.skew(xies))
#pandas.DataFrame.skew()
#statistics.pstdev() and statistics.pvariance()
#statsmodels.stats.descriptives.describe(): 
#scipy.stats.describe()


#Correlation_Pearson
#(n∑XY - ∑X∑Y) / sqrt([n∑X^2 - (∑X)^2][n∑Y^2 - (∑Y)^2])
def pearson_r(list1, list2):
    n = len(list1)
    sum_x = sum(list1)
    sum_y = sum(list2)
    sum_x_sq = sum([x**2 for x in list1])
    sum_y_sq = sum([y**2 for y in list2])
    sum_xy = sum([list1[i]*list2[i] for i in range(n)])
    numerator = n*sum_xy - sum_x*sum_y
    denominator = math.sqrt((n*sum_x_sq - sum_x**2) * (n*sum_y_sq - sum_y**2))
    if denominator == 0:
        return 0
    else:
        return numerator / denominator
#print(pearson_r(xies,yies))


#Correlation_Spearman
def spearman_r(list1, list2):
    sorted_lst1 = sorted(set(list1))
    ranks1 = [sorted_lst1.index(value) + 1 for value in list1]
    sorted_lst2 = sorted(set(list2))
    ranks2 = [sorted_lst2.index(value) + 1 for value in list2]

    n = len(ranks2)   
    up = 6 * sum((ranks1[i] - ranks2[i])**2 for i in range(n))
    down = n * (n**2 -1)
    rs = 1 - up/down
    return rs
#print(spearman_r(xies,yies))


#Correlation_Kendall
def kendall_tau(list1, list2):
    n = len(list1)
    assert n == len(list2)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i+1, n):
            if (list1[i] < list1[j] 
                and list2[i] < list2[j]) or (list1[i] > list1[j] 
                and list2[i] > list2[j]):
                concordant += 1
            else:
                discordant += 1
    tau = (concordant - discordant) / (0.5 * n * (n-1))
    return tau
#print(kendall_tau(xies,yies))


#Covariance
#Σ[(Xi - x̄) * (Yi - ȳ)] / (n - 1)
def covariance(list1, list2):
    av1 = average(list1)
    av2 = average(list2)
    x = [(a-av1) * (b-av2) for a, b in zip(list1, list2)]
    return sum(x)/(len(list1)-1)
#print(covariance(xies,yies))


#Regression
def reg(X,list1,list2):
    regression_coefficient = covariance(list1,list2)/variance(list1)
    Y = regression_coefficient * X
    return Y 
#print(reg(20,xies,yies))


#Slope
def slope(x1,x2,y1,y2):
    slope = (y2-y1)/(x2-x1)
    return slope
#print(slope(1,4,2,8))


#Intercept_point
def intercept(list1,list2):
    found_intersection = False
    for i in list1:
        for j in list2:
            if i == j:
                found_intersection = True
                break
        if found_intersection:
            break
    if not found_intersection:
        print("No intercept point")
#print(intercept(xies,yies))


#P-value
import statsmodels.api as sm
def calculate_p_value_statsmodels(sample1, sample2):
    # Calculate the t-statistic and p-value for a two-tailed independent t-test
    t_statistic, p_value, _ = sm.stats.ttest_ind(sample1, sample2)
    return p_value

from scipy.stats import ttest_ind
def calculate_p_value_scipy(sample1, sample2):
    # Calculate the t-statistic and p-value for a two-tailed independent t-test
    t_statistic, p_value = ttest_ind(sample1, sample2)
    return p_value

#ANOVA
def anova_3_lists(list1, list2, list3):
    # Calculate the group means
    mean1 = sum(list1) / len(list1)
    mean2 = sum(list2) / len(list2)
    mean3 = sum(list3) / len(list3)

    # Calculate the total mean
    total_mean = (sum(list1) + sum(list2) + sum(list3)) / (len(list1) + len(list2) + len(list3))

    # Calculate the sum of squares between groups
    ss_between = len(list1) * (mean1 - total_mean)**2 + len(list2) * (mean2 - total_mean)**2 + len(list3) * (mean3 - total_mean)**2

    # Calculate the sum of squares within groups
    ss_within = sum((x - mean1)**2 for x in list1) + sum((x - mean2)**2 for x in list2) + sum((x - mean3)**2 for x in list3)

    # Calculate the degrees of freedom
    df_between = 3 - 1
    df_within = len(list1) + len(list2) + len(list3) - 3

    # Calculate the mean square values
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    # Calculate the F-statistic and its associated p-value
    f_statistic = ms_between / ms_within
    p_value = stats.f.sf(f_statistic, df_between, df_within)

    return f_statistic, p_value



def anova_5_lists(list1, list2, list3, list4, list5):
    # Calculate the group means
    mean1 = sum(list1) / len(list1)
    mean2 = sum(list2) / len(list2)
    mean3 = sum(list3) / len(list3)
    mean4 = sum(list4) / len(list4)
    mean5 = sum(list5) / len(list5)

    # Calculate the total mean
    total_mean = (sum(list1) + sum(list2) + sum(list3) + sum(list4) + sum(list5)) / (len(list1) + len(list2) + len(list3) + len(list4) + len(list5))

    # Calculate the sum of squares between groups
    ss_between = len(list1) * (mean1 - total_mean)**2 + len(list2) * (mean2 - total_mean)**2 + len(list3) * (mean3 - total_mean)**2 + len(list4) * (mean4 - total_mean)**2 + len(list5) * (mean5 - total_mean)**2

    # Calculate the sum of squares within groups
    ss_within = sum((x - mean1)**2 for x in list1) + sum((x - mean2)**2 for x in list2) + sum((x - mean3)**2 for x in list3) + sum((x - mean4)**2 for x in list4) + sum((x - mean5)**2 for x in list5)

    # Calculate the degrees of freedom
    df_between = 5 - 1
    df_within = len(list1) + len(list2) + len(list3) + len(list4) + len(list5) - 5

    # Calculate the mean square values
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    # Calculate the F-statistic and its associated p-value
    f_statistic = ms_between / ms_within
    p_value = stats.f.sf(f_statistic, df_between, df_within)

    return f_statistic, p_value


#Kurtosis
def calculate_kurtosis(values):
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    m4 = sum((x - mean) ** 4 for x in values) / n
    kurtosis = (m4 / variance**2) - 3
    return kurtosis

'''
scipy.stats.kurtosis
numpy.kurtosis
pandas.DataFrame.kurtosis
statsmodels.stats.descriptives'''


#Chi squared test
def chi_squared_test(observed, expected=None):
    """
    Performs a chi-squared goodness of fit test.

    Args:
    - observed (list or array): A list or array of observed frequencies.
    - expected (list or array, optional): A list or array of expected frequencies. If not provided, assumes a uniform distribution.

    Returns:
    - chi_squared (float): The chi-squared test statistic.
    - p_value (float): The p-value of the test.
    """

    # If expected frequencies are not provided, assume a uniform distribution
    if expected is None:
        expected = [sum(observed) / len(observed)] * len(observed)

    # Calculate the chi-squared test statistic
    chi_squared = sum([(o - e)**2 / e for o, e in zip(observed, expected)])

    # Calculate the degrees of freedom
    df = len(observed) - 1

    # Calculate the p-value
    p_value = 1 - stats.chi2.cdf(chi_squared, df)

    return chi_squared, p_value

