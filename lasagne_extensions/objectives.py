from lasagne.objectives import *


def mae(x, t):
    """
    Calculates the MAE across all dimensions, i.e. feature
    dimension AND minibatch dimension.

    :param x: predicted values.
    :param t: target values.
    :return: the mean average error.
    """
    mae = t - x
    return (1. / mae.shape[0]) * abs(mae)


def mse(x, t):
    """
    Calculates the MSE across all dimensions, i.e. feature
    dimension AND minibatch dimension.

    :param x: predicted values.
    :param t: target values.
    :return: the mean average error.
    """
    return squared_error(x, t).mean(axis=1)


def sse(x, t):
    return squared_error(x, t).sum(axis=1)
