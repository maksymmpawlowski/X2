import numpy as np
import pandas as pd
import scipy
import statistics
import math

xies = [1,2,3]
yies = [2,4,6]

def covariance_gpt(x, y):
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
#print("gpt's:", covariance_gpt(xies,yies))


def cov(l1,l2):
    n = len(l1)
    av1 = sum(l1)/n
    av2 = sum(l2)/n
    multisum = 0
    for i in range(n):
        multisum += ((l1[i]-av1)) * ((l2[i]-av2))
    cova = multisum/(n-1)
    return cova
print("mine2:",cov(xies,yies))


'''
Could you write for me a python code 
multiplying pairs of two lists elements, 
where from every element of pair has been 
subtracted the mean of its list?
'''
list1 = [1,2,3]
list2 = [2,4,6]
n = len(list1)
mean1 = sum(list1) / n
mean2 = sum(list2) / n
result = []
for i, j in zip(list1, list2):
    result.append((i - mean1) * (j - mean2))
r = sum(result)
n = n - 1
r = sum(result)/n
print(r)




#pandas 
df = pd.DataFrame({'x': xies, 'y': yies})
cov = df[['x', 'y']].cov().iloc[0, 1]
#print(cov)
'''
l1 = ["a","b", "c"]
l2 = ["a","b", "c"]
df = pd.DataFrame({'x': l1, 'y': l2})
cov = df[['x', 'y']].cov().iloc[0, 1]
print(cov)
'''
x = pd.Series(xies)
y = pd.Series(yies)
cov = x.cov(y)
#print(cov)
'''
This function computes the pairwise correlation 
among the columns of a DataFrame. Since 
covariance is related to correlation by a scaling 
factor, you can use this function to compute the 
covariance as well. You can pass the two columns 
containing x and y as arguments to this function 
and multiply the result by the product of the 
standard deviations of x and y. For example:
'''
df = pd.DataFrame({'x': xies, 'y': yies})
cov = df[['x', 'y']].corr().iloc[0, 1] * df['x'].std() * df['y'].std()
#print(cov)














#numpy
'''
numpy.cov(): This function from the NumPy library can be used to calculate the covariance matrix of a given set of variables.

pandas.DataFrame.cov(): This function from the Pandas library can be used to calculate the pairwise covariance between columns of a DataFrame.

scipy.stats.covariance(): This function from the SciPy library can be used to calculate the covariance matrix of a given set of variables.

statistics.pstdev() and statistics.pvariance(): These functions from the statistics module in Python can be used to calculate the population standard deviation and variance, respectively, which can be used to calculate covariance.

math.fsum(): This function from the math module can be used to calculate the sum of a list of numbers, which can be used to calculate covariance.

statistics.pvariance() and statistics.covariance() - These functions are part of the statistics module in the Python standard library and can be used to calculate the population variance and covariance, respectively, of a set of variables.
'''

args = xies, yies
'''
f_dict = {
    "numpy 1-st" : np.cov(args),
    "pandas" : pd.DataFrame.cov(args),
    "scipy" : scipy.stats.covariance(args),
    "statistics" : statistics.pstdev(args),
    "math" : math.fsum(args)
}
for key,value in f_dict.items():
    print(key, "\t", value, "\n")
'''
'''
print(np.cov(args),"\n")
print(np.cov(xies, yies, bias=True),"\n")
print(np.cov(xies, yies, bias=False),"\n")
print(np.corrcoef(xies, yies, rowvar=True),"\n")
'''



'''
#print("mine:", covariance(xies,yies))
print("numpy:", np.cov(xies,yies))
#print("pandas", pd.Series.cov(xies,yies))
#print("pandas", pd.DataFrame.cov(xies,yies))
#print("scipy", scipy.stats.covariance(xies,yies))
#print("statistics:", statistics.pvariance(xies,yies))
print("numpy:", np.corrcoef(xies,yies))
'''





'''
def covariance(list1, list2):
    av1 = average(list1)
    av2 = average(list2)
    n = len(list1)
    x = [(a-av1) * (b-av2) for a, b in zip(list1, list2)]
    up = sum(x)
    down = n-1
    s = up/down
    return(s)
print(covariance(xies,yies))

def covariance4(l1,l2):
    av1 = average(l1)
    av2 = average(l2)
    n = len(l1)
    multisum = 0
    for i in range(n):
        multisum += ((l1[i]-av1)) * ((l2[i]-av2))
    cova = multisum/(n-1)
    return cova
print(covariance4(xies,yies))

def cov6(l1,l2):
    n = len(l1)
    av1 = sum(l1)/n
    av2 = sum(l2)/n
    multisum = 0
    for i in range(n):
        multisum += ((l1[i]-av1)) * ((l2[i]-av2))
    cova = multisum/(n-1)
    return cova
print(cov6(xies,yies))

def cov7(list1,list2):
    n = len(list1)
    mean1 = sum(list1) / n
    mean2 = sum(list2) / n
    result = []
    for i, j in zip(list1, list2):
        result.append((i - mean1) * (j - mean2))
    r = sum(result)
    n = n - 1
    return sum(result)/n
print(cov7(xies,yies))
'''