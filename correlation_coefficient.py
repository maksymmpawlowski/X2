import math
import numpy as np
import scipy
import pandas as pd
#pearson
#(n∑XY - ∑X∑Y) / sqrt([n∑X^2 - (∑X)^2][n∑Y^2 - (∑Y)^2])
xies = [1,24,1,513,46,1313,613,63]
yies = [745,2532,3,552,7272274,24,32,23]

def corcoe(list1,list2):
    squares1 = [item**2 for item in list1]
    squares2 = [item**2 for item in list2]
    multis12 = [a * b for a, b in zip(list1, list2)]

    sum1 = sum(list1)
    sum2 = sum(list2)
    sumultis = sum(multis12)
    sumsqrs1 = sum(squares1)
    sumsqrs2 = sum(squares2)
    sqrsum1 = sum1**2
    sqrsum2 = sum2**2
    n = len(xies)

    up = (n*sumultis) - (sum1*sum2)
    down = (n*sumsqrs1 - sqrsum1)*(n*sumsqrs2 - sqrsum2)
    down = math.sqrt(down)
    r = up/down
    return r

#print(corcoe(xies,yies))

#1.numpy
'''
x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 4, 3, 2, 1])
corr_coef, p_value = np.corrcoef(x, y)[0, 1]
print("Pearson correlation coefficient:", corr_coef)
print("p-value:", p_value)
'''
#2.scipy
'''
from scipy.stats import pearsonr

corr_coef, p_value = pearsonr(xies, yies)
print("Pearson r:", corr_coef)
print("p-value:", p_value)
'''

#3.pandas
'''
df = pd.DataFrame({'x': xies, 'y': yies})
#Get the Pearson r between columns x and y
corr_coef = df['x'].corr(df['y'])
print(corr_coef)'''

#4.statsmodels.api
'''
import statsmodels as sm
from statsmodels import statsmodels
corr_coef, p_value = sm.stats.pearsonr(xies, yies)
print(corr_coef,p_value)'''




#spearman
'''
can you write me a python function that takes 
lists as arguments and that returns new lists 
containing the ranks assigned to each element 
of the lists that are arguments to this function?

def rank_lists(*lists):
    
    Assigns ranks to elements of input lists and returns new lists with ranks.
    
    Args:
        *lists: One or more input lists
        
    Returns:
        A tuple of new lists, where each list contains the ranks assigned to the 
        elements of the corresponding input list.
    
    # Combine all input lists into a single list
    combined_list = []
    for lst in lists:
        combined_list.extend(lst)
    
    # Sort the combined list in ascending order
    sorted_list = sorted(combined_list)
    
    # Assign ranks to each element in the sorted list
    ranks = {}
    rank = 1
    for elem in sorted_list:
        if elem not in ranks:
            ranks[elem] = rank
            rank += 1
    
    # Create new lists containing ranks for each input list
    output_lists = []
    for lst in lists:
        output_lists.append([ranks[elem] for elem in lst])
    
    # Return the new lists
    return tuple(output_lists)

'''


def sum_of_squared_differences(list1, list2):
    #Calculates the sum of squared differences between two lists
    if len(list1) != len(list2):
        return None  # return None if lists are not of the same length
    else:
        return sum([(list1[i] - list2[i])**2 for i in range(len(list1))])


def spearman_r(list1, list2):
    #assign the ranks to lists
    sorted_lst1 = sorted(set(list1))
    ranks1 = [sorted_lst1.index(value) + 1 for value in list1]
    sorted_lst2 = sorted(set(list2))
    ranks2 = [sorted_lst2.index(value) + 1 for value in list2]
    #sum(([abs(x-y) for x,y in zip(list1,list2)]))

    n = len(ranks2)   
    up = 6 * sum_of_squared_differences(ranks1,ranks2)
    down = n * (n**2 -1)
    rs = 1 - up/down
    return rs
print("my spearman: " + str(spearman_r(xies,yies)))

#scipy.stats
import scipy.stats as stats
rho, p_value = stats.spearmanr(xies, yies)
print("Spearman correlation coefficient:", rho)

#pandas
df = pd.DataFrame({'x': xies, 'y': yies})
rho = df.corr(method='spearman').iloc[0, 1]
print("Spearman correlation coefficient:", rho)

#numpy 
rho = np.corrcoef(xies, yies, rowvar=False)[0, 1]
print("Spearman correlation coefficient:", rho)

def pearson_correlation(list1, list2):
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
print("gpt's spearman: "+str(pearson_correlation(xies,yies)))

def spearman_rr(list1, list2):
    #assign the ranks to lists
    sorted_lst1 = sorted(set(list1))
    ranks1 = [sorted_lst1.index(value) + 1 for value in list1]
    sorted_lst2 = sorted(set(list2))
    ranks2 = [sorted_lst2.index(value) + 1 for value in list2]

    n = len(ranks2)   
    up = 6 * sum((ranks1[i] - ranks2[i])**2 for i in range(n))
    down = n * (n**2 -1)
    rs = 1 - up/down
    return rs
print("upgraded spearman :"+str(spearman_rr(xies,yies)))
