from basic_analysis import average, deviation
import numpy as np
import pandas as pd
import scipy
import statistics
xies = [1,2,3]
yies = [2,4,6]
#sXY = Σ[(Xi - x̄) * (Yi - ȳ)] / (n - 1)
#Cov(X,Y) = Σ[(Xi - μX) * (Yi - μY)] / N
def covariance(list1, list2):
    av1 = average(list1)
    av2 = average(list2)
    n = len(list1)
    x = [(a-av1) * (b-av2) for a, b in zip(list1, list2)]
    up = sum(x)
    down = n-1
    s = up/down
    return(s)

#print("mine:", covariance(xies,yies))
#print("numpy:", np.cov(xies,yies))
#print("pandas", pd.Series.cov(xies,yies))
#print("pandas", pd.DataFrame.cov(xies,yies))
#print("scipy", scipy.stats.covariance(xies,yies))
#print("statistics:", statistics.pvariance(xies,yies))
###print("numpy:", np.corrcoef(xies,yies))

def covariance_coefficient(x, y):
    n = len(x)
    if n != len(y):
        raise ValueError("The length of the two lists must be the same.")
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    cov /= n-1
    std_x = (sum((xi - mean_x) ** 2 for xi in x) / (n-1)) ** 0.5
    std_y = (sum((yi - mean_y) ** 2 for yi in y) / (n-1)) ** 0.5
    return cov / (std_x * std_y)
print("gpt's:", covariance_coefficient(xies,yies))




mean1 = sum(xies) / len(xies)
mean2 = sum(yies) / len(yies)
result = []
for x, y in zip(xies, yies):
    result.append((x - mean1) * (y - mean2))
#print(result)


#cov(X,Y) = (1/n) * Σ[(xi-μx) * (yi-μy)]
def cov2(l1,l2):
    av1 = average(l1)
    av2 = average(l2)
    n = len(l1)
    multisum = 0
    for item in range(n):
        multisum += (l1[item]-av1) * (l2[item]-av2)
    cvv = multisum/n
    return cvv
#print(cov2(xies,yies))

'''
def multiply_sum(list1, list2):
    result = 0
    for i in range(len(list1)):
        result += list1[i] * list2[i]
    return result
print(multiply_sum(xies,yies))
'''

#Cov(X,Y) = E[(X - E[X])(Y - E[Y])]
def cov(l1,l2):
    av1 = average(l1)
    av2 = average(l2)
    n = len(l1)
    multisum = 0
    for i in range(n):
        multisum += ((l1[i]-av1)) * ((l2[i]-av2))
    cova = multisum/(n-1)
    return cova
#print("mine2:",cov(xies,yies))




def covariance_coefficient0(x, y):
    n = len(x)
    if n != len(y):
        raise ValueError("Lists must have the same length")
    
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    cov_xy = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    cov_xy /= n - 1
    
    std_dev_x = (sum((x[i] - mean_x)**2 for i in range(n)) / (n - 1))**0.5
    std_dev_y = (sum((y[i] - mean_y)**2 for i in range(n)) / (n - 1))**0.5
    
    return cov_xy / (std_dev_x * std_dev_y)
