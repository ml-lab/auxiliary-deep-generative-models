import math
import theano.tensor as T
import numpy as np
# library with theano PDF functions
c = - 0.5 * math.log(2*math.pi)
def normal(x, mean, sd):
	return c - T.log(T.abs_(sd)) - (x - mean)**2 / (2 * sd**2)

def normal2(x, mean, logvar):
	return c - logvar/2 - (x - mean)**2 / (2 * T.exp(logvar))

def laplace(x, mean, logvar):
    sd = T.exp(0.5 * logvar)
    return - abs(x - mean) / sd - 0.5 * logvar - np.log(2)

def standard_normal(x):
	return c - x**2 / 2

# Centered laplace with unit scale (b=1)
def standard_laplace(x):
	return math.log(0.5) - T.abs_(x)