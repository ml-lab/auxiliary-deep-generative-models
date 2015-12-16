import numpy as np
import theano.tensor as T
import lasagne
from lasagne.layers.base import Layer
import math


class GaussianLogDensityLayer(lasagne.layers.MergeLayer):
    def __init__(self, x, mu, var, **kwargs):
        self.x, self.mu, self.var = None, None, None
        if not isinstance(x, Layer):
            self.x, x = x, None
        if not isinstance(mu, Layer):
            self.mu, mu = mu, None
        if not isinstance(var, Layer):
            self.var, var = var, None
        input_lst = [i for i in [x, mu, var] if not i is None]
        super(GaussianLogDensityLayer, self).__init__(input_lst, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, input, **kwargs):
        x = self.x if self.x is not None else input.pop(0)
        mu = self.mu if self.mu is not None else input.pop(0)
        logvar = self.var if self.var is not None else input.pop(0)

        c = - 0.5 * math.log(2 * math.pi)
        density = c - logvar / 2 - (x - mu) ** 2 / (2 * T.exp(logvar))
        return T.sum(density, axis=-1, keepdims=True)


class GaussianMarginalLogDensityLayer(lasagne.layers.MergeLayer):
    def __init__(self, mu, var, **kwargs):
        self.mu, self.var = None, None
        if not isinstance(mu, Layer):
            self.mu, mu = mu, None
        if not isinstance(var, Layer):
            self.var, var = var, None
        input_lst = [i for i in [mu, var] if not i is None]
        super(GaussianMarginalLogDensityLayer, self).__init__(input_lst, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, input, **kwargs):
        mu = self.mu if self.mu is not None else input.pop(0)
        logvar = self.var if self.var is not None else input.pop(0)

        if mu == 1:
            density = -0.5 * (T.log(2 * np.pi) + 1 + logvar)
        else:
            density = -0.5 * (T.log(2 * np.pi) + (T.sqr(mu) + T.exp(logvar)))
        return T.sum(density, axis=-1, keepdims=True)


class BernoulliLogDensityLayer(lasagne.layers.MergeLayer):
    def __init__(self, x_mu, x, eps=1e-6, **kwargs):
        input_lst = [x_mu]
        self.eps = eps
        self.x = None

        if not isinstance(x, Layer):
            self.x, x = x, None
        else:
            input_lst += [x]
        super(BernoulliLogDensityLayer, self).__init__(input_lst, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, input, **kwargs):
        x_mu = input.pop(0)
        x = self.x if self.x is not None else input.pop(0)

        if x_mu.ndim > x.ndim:  # Check for sample dimensions.
            x = x.dimshuffle((0, 'x', 'x', 1))

        x_mu = T.clip(x_mu, self.eps, 1 - self.eps)
        density = T.sum(-T.nnet.binary_crossentropy(x_mu, x), axis=-1, keepdims=True)
        return density


class MultinomialLogDensityLayer(lasagne.layers.MergeLayer):
    def __init__(self, x_mu, x, eps=1e-8, **kwargs):
        input_lst = [x_mu]
        self.eps = eps
        self.x = None
        if not isinstance(x, Layer):
            self.x, x = x, None
        else:
            input_lst += [x]
        super(MultinomialLogDensityLayer, self).__init__(input_lst, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, input, **kwargs):
        x_mu = input.pop(0)
        x = self.x if self.x is not None else input.pop(0)

        # Avoid Nans
        x_mu += self.eps

        if x_mu.ndim > x.ndim:  # Check for sample dimensions.
            x = x.dimshuffle((0, 'x', 'x', 1))

        density = -(-T.sum(x * T.log(x_mu), axis=-1, keepdims=True))
        return density
