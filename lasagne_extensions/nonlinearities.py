__author__ = 'larsma'

from lasagne.nonlinearities import *
import theano.tensor as T

def softplus(x): return T.log(T.exp(x) + 1)

import lasagne
lasagne.layers.InputLayer
l_in = lasagne.layers.InputLayer(shape=(None, 784))
l_hid = lasagne.layers.DenseLayer(l_in, num_units=500, nonlinearity=softplus)
l_out = lasagne.layers.DenseLayer(l_hid, num_units=10, nonlinearity=lasagne.nonlinearities.softmax)