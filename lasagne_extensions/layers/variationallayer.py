import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers.base import Layer
import math
from theano.tensor.shared_randomstreams import RandomStreams


class GaussianSampleLayer(lasagne.layers.MergeLayer):
    """
    Used for sampling in VAE
    I you specify random_sym you are responsible for passing in the
    random numbers used for drawning samples.
    """
    def __init__(self, mu, var, nsamples=1, random_sym = None,**kwargs):
        super(GaussianSampleLayer, self).__init__([mu, var], **kwargs)

        self.samples = nsamples
        self.random_sym = random_sym

        if random_sym is None:
            self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, input, deterministic=False, **kwargs):
        mu, log_var = input
        # if deterministic:
        #     return mu
        r, c = mu.shape
        if self.random_sym is None:
            eps = self._srng.normal((r, self.samples, c))
        else:
            if not self.random_sym.shape[0] == self.samples:
                raise NotImplementedError()
            eps = self.random_sym

        z = mu.dimshuffle(0, 'x', 1) + T.exp(0.5 * log_var.dimshuffle(0, 'x', 1)) * eps
        z = z.reshape((-1, c))
        return z

class IWGaussianSampleLayer(lasagne.layers.MergeLayer):
    """
    Samplelayer supporting importance sampling as described in [BURDA]_ and
    multiple monte carlo samples for the approximation of
    E_q [log( p(x,z) / q(z|x) )]
    Parameters
    ----------
    mu, log_var : class:`Layer` instances
        Parameterizing the mean and log(variance) of the distribution to sample
        from as described in [BURDA]. The code assumes that these have the same
        number of dimensions
    eq_samples: Int or T.scalar
        Number of Monte Carlo samples used to estimate the expectation over
        q(z|x) in eq. (8) in [BURDA]
    iw_samples: Int or T.scalar
        Number of importance samples in the sum over k in eq. (8) in [BURDA]
    References
    ----------
        ..  [BURDA] Burda, Yuri, Roger Grosse, and Ruslan Salakhutdinov.
            "Importance Weighted Autoencoders."
            arXiv preprint arXiv:1509.00519 (2015).
    """

    def __init__(self, mu, log_var, eq_samples=1, iw_samples=1, **kwargs):
        super(IWGaussianSampleLayer, self).__init__([mu, log_var], **kwargs)

        self.eq_samples = eq_samples
        self.iw_samples = iw_samples

        self._srng = RandomStreams(
            lasagne.random.get_rng().randint(1, 2147462579))

    def get_output_shape_for(self, input_shapes):
        batch_size, num_latent = input_shapes[0]
        if isinstance(batch_size, int) and \
           isinstance(self.iw_samples, int) and \
           isinstance(self.eq_samples, int):
            out_dim = (batch_size*self.eq_samples*self.iw_samples, num_latent)
        else:
            out_dim = (None, num_latent)
        return out_dim

    def get_output_for(self, input, **kwargs):
        mu, log_var = input
        batch_size, num_latent = mu.shape
        eps = self._srng.normal(
            [batch_size, self.eq_samples, self.iw_samples, num_latent],
             dtype=theano.config.floatX)

        z = mu.dimshuffle(0,'x','x',1) + \
            T.exp(0.5 * log_var.dimshuffle(0,'x','x',1)) * eps

        return z.reshape((-1,num_latent))


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

        c = - 0.5 * math.log(2*math.pi)
        density = c - logvar/2 - (x - mu)**2 / (2 * T.exp(logvar))
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

        if x_mu.ndim == 3 and x.ndim == 2:
            x = x.dimshuffle((0, 'x', 1))
        elif x_mu.ndim == 4 and x.ndim == 2:
            x = x.dimshuffle((0, 'x', 'x', 1))

        x_mu = T.clip(x_mu, self.eps, 1-self.eps)
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

        if x_mu.ndim == 3 and x.ndim == 2:
            x = x.dimshuffle((0, 'x', 1))
        elif x_mu.ndim == 4 and x.ndim == 2:
            x = x.dimshuffle((0, 'x', 'x', 1))

        density = -(-T.sum(x * T.log(x_mu), axis=-1, keepdims=True))
        return density