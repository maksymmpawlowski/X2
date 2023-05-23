import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
import statistics
import pandas as pd

'''
accomodation assistant
house keeping
room attendant
list of hotels: 
- The Doyle collection/The Westbury
- The fleet hotel
- Morrison Hotel
- Morgan Hotel
- The Spencer Hotel
- Hard Rock Hotel
- Celtic Lodge Guesthouse
- Wren Urban Nest


'''
#what is this 
#simple function counting it
#when to do it 

#list or *args?

#Arithmetic average:
def average(list):
    arithmetic_average = sum(list)/len(list)
    return arithmetic_average

#Median:
def mediann(list):
    list.sort()
    centre = len(list)/2
    if len(list) % 2 == 0:
        p1 = list[int(centre)]
        p2 = list[int(centre-1)]
        pp = (p1+p2)/2
        return pp
    else:
        return list[int(centre)]

#Variance
'''Measure of volatility/diversity'''
def variance(list):
    list2 = []
    for item in list:
        x = (pow(item - average(list), 2))
        list2.append(x)
    return sum(list2)/len(list)

'''https://builtin.com/data-science/how-to-find-the-variance'''
def calculate_variance(population, is_sample = False):
   #calculate the mean
   mean = (sum(population) / len(population))
   #calculate differences
   diff = [(v - mean) for v in population]
   #Square differences and sum
   sqr_diff = [d**2 for d in diff]
   sum_sqr_diff = sum(sqr_diff)
   #calculate variance
   if is_sample == True:
       variance = sum_sqr_diff/(len(population) - 1)
   else:
       variance = sum_sqr_diff/(len(population))
   return variance

#Standard deviation
'''
Average data distance around the mean

It shows how far, on average, all elements 
of the set are from the average value. 
What does it matter if the average itself 
is crap and doesn't carry any reliable 
information? What's the point of counting 
the average deviation in the set 
[1,2,3,4,10000] at all??? If the collection 
is shit, there's no point in describing the 
shit
'''
def deviation(list):
    devi = np.sqrt(variance(list))
    return devi

#Percentile
'''
percentile X is the number from which X% of 
the elements of our set are smaller
np.percentile(xies,23)
((90/100) * len(xies))
'''
def percentile00(list, num):
    sortlist = sorted(list)
    x = ((len(sortlist)-1)*num)/100
    y = round(x)
    z = x - y
    if y + 1 < len(sortlist):
        return sortlist[y] + z * (sortlist[y+1] - sortlist[y])
    else:
        return sortlist[y]


def percentile1(list, num):
    list.sort()
    x = ((len(list)-1)*num)/100
    y = round(x) 
    #the rest from rounding
    z = x - y
    if y + 1 < len(list):
        return list[y]+z*(list[y+1] - list[y])
    else:
        return list[y]
    
xies = [12,129,12,51,6,3,78]
#print(percentile1(xies, 90))

#absolute value
#abs(num)
xies = [3, 7, 9, 12, 15, 18, 21, 26, 30, 36]

def percentile0(list, num):
    list.sort()
    lenn = len(list)
    index = (num*lenn)/100
    i = round(index)
    d = index-i
    val = (1-d)*list[i-1] + d*list[i]
    return val
#print(percentile(xies, 75))

def percentile(list, num):
    list.sort()
    lenn = len(list)
    r = (num/100)*(lenn+1)
    
#print(percentile(xies, 75))

#percentile general pattern
xies = [3, 5, 7, 8, 10, 12, 15, 18, 20, 22]
def general_percentile(list,num):
    n = len(list)
    f = num
    P = (n + 1)*f/100
    return P
#print(general_percentile(xies, 30))
'''
Since the resulting value is not a whole number, 
it represents an interpolated value between the 
3rd and 4th values in the sorted dataset, which 
are 7 and 8, respectively. Therefore, the 30th 
percentile of the dataset is 7.3 (rounded to one 
decimal place).
'''

'''
print(
    percentile00(xies, 30),
    percentile(xies, 30),
    percentile1(xies, 30),
    percentile0(xies, 30),
    np.percentile(xies,30)
    )
'''


'''
list of all the python functions that compute 
the exact percentile value of a set, along with 
the percentile counting method they use:
'''
set = [
    412,21,5,1221,2521,0.00005,3,5,26,312,16,
    12,512,0.0002153,633,615,21,31,87,8,6545,
    21332,0.000316,21,2521,327,43,282,4,2,73,
    57,3,3,3232,787,0.000042,1292,157,1,91,81
    ]
#1.	numpy.percentile
'''
The interpolation method which can be either 
'linear', 'lower', 'higher', 'midpoint', 
or 'nearest' '''
print(np.percentile(set,30))
#2.scipy.stats.scoreatpercentile
'''
Percentile ranking method.'''
print(scipy.stats.scoreatpercentile(set,30),'\n')

#3.pandas.Series.quantile
'''
Linear interpolation method by default, 
but this can be changed by setting the 
interpolation parameter to either 'lower', 
'higher', 'midpoint', or 'nearest'.'''
series = pd.Series(set)
print(pd.Series.quantile(series,[0.3]),'\n')
#4.	statistics.quantiles 
'''The nearest-rank method. Returns a list of 
quantiles, where q is a list of quantile values'''
print(statistics.quantiles(set, n=3),'\n')
#5.numpy.nanpercentile
'''Computing along the specified axis, 
ignoring NaN values. Interpolation method which
can be either 'linear', 'lower', 'higher', 
'midpoint', or 'nearest' '''
#https://www.geeksforgeeks.org/numpy-nanpercentile-in-python/
#6.	numpy.quantile
'''The nearest-rank method. Returns a list of 
quantiles, where q is a list of quantile 
values.'''
#print(np.quantile(set,3))

'''percentile, decile, quartile, quantile'''

'''histogram generator'''

'''boxplot generator'''

'''covariance'''

#Regression
'''Counting correlation between two variables
and how one is changing to another'''

#General pattern for linear regression
'''
xies = [3,3,2,4,1]
yies = [20,25,20,30,10]
xsquares = []
for x in xies:
    xsquares.append(x**2)
multixy = [a * b for a, b in zip(xies, yies)]
n = len(xies)

sumxies = sum(xies)
sumyies = sum(yies)
multisum = sum(multixy)
sumxsquares = sum(xsquares)

up = (n*multisum) - (sumxies*sumyies)
down = (n*sumxsquares) - (sumxies**2)
regression_parameter = up/down
print(regression_parameter)

b = (n * sum(multixy) - sum(xies) * sum(yies))/(n * sum(xsquares) - sum(xies)**2)
print(b)
a = arverage(yies) - b * arverage(xies) 
y = a + b * x

print(y)
'''
#add elements of two lists
#result = [a * b for a, b in zip(xies, yies)]










#Krzysztof 0877951446
#Correlation coefficent(r)
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

'''
set1 = [1, 2, 3, 4, 5]
set2 = [2, 4, 6, 8, 10]
s1 = [1,2,3,4,5,6]
s2 = [1,4,9,16,25,36]
s3 = [2,4,6,8,10,12]

num = 1
list = [
    #(corcoe(s1,s2)),
    #(corcoe(s1,s3)),
    (corcoe(xies,yies)),
    (corcoe(set1,set2)),
    #(corcoe(set2,set1)),
    (np.corrcoef(xies, yies)),
    (np.corrcoef(set1, set2)),
    (np.corrcoef(set1, set2)[0,1]),

    (np.corrcoef(xies, y=None)),
    (np.corrcoef(yies, y=None)),
    (np.corrcoef(set1, y=None)),
    (np.corrcoef(set2, y=None))
    ]   
for item in list:
    print(num,'\t', item,'\n')
    num +=1

x = np.random.rand(100)
y = np.random.rand(100)
corrcoef_matrix = np.corrcoef(x, y)
print(corrcoef_matrix)
print("\n",corcoe(x,y))
'''
#calculate the square root of
#print(np.sqrt(4564288385264006651924))
'''Why in python can we calculate the square 
root of the number 4564288385264006651924 using 
the math module but not using the numpy module?'''
'''
In Python, the math module and the numpy module 
both provide functions for calculating the square 
root of a number. However, the difference between 
the two modules lies in the range of numbers they can handle.

The square root of the number 4564288385264006651924 
is a very large number, which exceeds the maximum 
value that can be represented by a 64-bit floating point number.

The math module uses the standard floating-point 
arithmetic provided by the CPU, which can handle 
numbers up to a certain range before rounding 
errors start to accumulate. It can handle the 
square root of the number 4564288385264006651924 
because it returns an approximation of the result 
within the limits of its floating-point arithmetic.

On the other hand, the numpy module uses a specialized 
array data type called numpy.ndarray that can handle very 
large numbers without rounding errors. However, the numpy.sqrt() 
function returns a nan (not-a-number) value when trying to calculate 
the square root of a number that is too large to represent as a 64-bit 
floating point number. This is because the numpy.sqrt() function requires 
that the input be within the range of representable numbers for a 
64-bit floating point number.

Therefore, while the math module can handle the square root 
of the number 4564288385264006651924 by returning an approximation, 
the numpy module returns nan because the number is outside of the 
representable range.
'''


#Dominants
'''
The value of the variable feature 
that occurs most often in a given population
'''

x = [1,1,1,1,122,252,12,2,12,123,122,252,12,12,12]
def dominants(set):
    list = []
    list0 = []
    dict = {}
    for x in set:
        list0.append(x)
        if x not in list:
            list.append(x)
    list.sort()
    for x in list:
        dict[x] = list0.count(x)
    for key, value in dict.items():
        print(key, "\t",value)


def count_elements(lst):
    """Counts the occurrences of each element in a list and returns a dictionary."""
    element_counts = {}
    for element in lst:
        if element in element_counts:
            element_counts[element] += 1
        else:
            element_counts[element] = 1
    #new_dict = {str(key): str(value) for key, 
                #value in element_counts.items()}
    sorted_dict = dict(sorted(element_counts.items(), key=lambda item: item[1], reverse=True))
    return sorted_dict
'''
To display only the 3 elements of a dictionary 
with the greatest keys, you can use the sorted() 
function with the reverse parameter set to True 
to sort the dictionary by keys in descending 
order, and then use a slice to get the first 
three elements. Here is an example:

my_dict = {'apple': 5, 'banana': 8, 'cherry': 3, 'date': 9, 'elderberry': 1}

In this example, we have a dictionary my_dict 
with keys and values representing fruit names and 
their quantities. We sort the dictionary by keys 
in descending order using the sorted() function 
and the reverse=True parameter. Then, we create a 
new dictionary top_three by slicing the sorted 
dictionary and getting only the first three 
elements using the list() function and a slice 
[0:3]. Finally, we print out the top_three 
dictionary, which will contain the 3 elements 
with the greatest keys.


# sort the dictionary by keys in descending order
sorted_dict = dict(sorted(my_dict.items(), reverse=True))
# by values
sorted_dict = dict(sorted(my_dict.items(), key=lambda item: item[1]))
# get the first three elements
top_three = dict(list(sorted_dict.items())[:3])

print(top_three)  # {'elderberry': 1, 'date': 9, 'cherry': 3}
'''

'''
To display the keys and values of a dictionary in 
Python sorted by values, you can use the sorted() 
function along with a lambda function to sort the 
dictionary by its values.

my_dict = {'apple': 3, 'banana': 2, 'orange': 5, 'pear': 1}

# Sort the dictionary by values using the sorted() function and a lambda function
sorted_dict = dict(sorted(my_dict.items(), key=lambda item: item[1]))

# Print the sorted dictionary
for key, value in sorted_dict.items():
    print(key, value)

In this example, the sorted() function is used to sort the dictionary my_dict by its values. The key parameter of the sorted() function is set to a lambda function that returns the value of each item in the dictionary. The items() method of the dictionary is used to convert it into a list of key-value pairs, which is then sorted by the lambda function. The sorted list is then converted back into a dictionary using the dict() function.

Finally, the keys and values of the sorted dictionary are printed using a for loop.

'''
'''
print(count_elements(x))
#def intervals_dominants(list):
for key, value in dict(list(count_elements(x).items())[:3]).items():
#dict(list(count_elements(x).items())[:3])  
    print(key,"\t",value)
'''

