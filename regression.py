from covariance import covariance
from basic_analysis import variance, average
import numpy as np
import scipy

xies = [1,2,3,4]
yies = [2,4,6,8]
def regression(xies, yies):
    b1 = covariance(xies,yies) / variance(xies)
    b0 = average(yies) - b1 * average(yies)
    return b1, b0

print("mine:",regression(xies,yies))
print("numpy:",np.polyfit(xies,yies,deg=1))
print("scipy:", scipy.stats.linregress(xies,yies))
#sklearn.linear_model.LinearRegression() 

