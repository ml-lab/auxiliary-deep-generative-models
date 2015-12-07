__author__ = 'larsma'

import os
import gzip
import cPickle
import numpy as np
from utils import env_paths

def _download():
    """
    Download the MNIST dataset if it is not present.
    :return: The train, test and validation set.
    """
    dataset = 'mnist.pkl.gz'
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            env_paths.get_data_path(),
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set, test_set, valid_set

def _pad_targets(xy):
    """
    Pad the targets to be 1hot.
    :param xy: A tuple containing the x and y matrices.
    :return: The 1hot coded dataset.
    """
    x, y = xy
    classes = np.max(y)+1
    tmp_data_y = np.zeros((x.shape[0], classes))
    for i, dp in zip(range(len(y)), y):
        r = np.zeros(classes)
        r[dp] = 1
        tmp_data_y[i] = r
    y = tmp_data_y
    return x, y

def _create_semi_supervised(xy, n_labeled, rng):
    """
    Divide the dataset into labeled and unlabeled data.
    :param xy: The training set of the mnist data.
    :param n_labeled: The number of labeled data points.
    :param rng: NumPy random generator.
    :return: labeled x, labeled y, unlabeled x, unlabeled y.
    """
    x, y = xy

    def _split_by_class(x, y, num_classes):
        result_x = [0]*num_classes
        result_y = [0]*num_classes
        for i in range(num_classes):
            idx_i = np.where(y == i)[0]
            result_x[i] = x[:,idx_i]
            result_y[i] = y[idx_i]
        return result_x, result_y

    x, y = _split_by_class(x.T, y.T, 10)

    def binarize_labels(y, n_classes=10):
        new_y = np.zeros((n_classes, y.shape[0]))
        for i in range(y.shape[0]):
            new_y[y[i], i] = 1
        return new_y

    for i in range(10):
        y[i] = binarize_labels(y[i])

    n_classes = y[0].shape[0]
    if n_labeled % n_classes != 0:
        raise("n_labeled (wished number of labeled samples) not divisible by n_classes (number of classes)")
    n_labels_per_class = n_labeled/n_classes
    x_labeled = [0]*n_classes
    x_unlabeled = [0]*n_classes
    y_labeled = [0]*n_classes
    y_unlabeled = [0]*n_classes
    for i in range(n_classes):
        idx = range(x[i].shape[1])
        rng.shuffle(idx)
        x_labeled[i] = x[i][:, idx[:n_labels_per_class]]
        y_labeled[i] = y[i][:, idx[:n_labels_per_class]]
        x_unlabeled[i] = x[i]
        y_unlabeled[i] = y[i]
    return np.hstack(x_labeled).T, np.hstack(y_labeled).T, np.hstack(x_unlabeled).T, np.hstack(y_unlabeled).T

def load_semi_supervised(n_batches=100, n_labeled=100, n_samples=100, filter_std=0.1, seed=123456, train_valid_combine=False):
    """
    Load the mnist dataset where only a fraction of data points are labeled. The amount
    of labeled data will be evenly distributed accross classes.
    :param n_batches: number of batches.
    :param n_labeled: number of labeled data points.
    :param n_samples: number of labeled samples for each batch.
    :param filter_std: the standard deviation threshold for keeping features.
    :param seed: the seed for the pseudo random shuffle of data points.
    :param train_valid_combine: if the train set and validation set should be combined.
    :return: train set, test set, validation set.
    """
    train_set, test_set, valid_set = _download()
    # Combine the train set and validation set.
    if train_valid_combine:
        train_set = np.append(train_set[0], valid_set[0], axis=0), np.append(train_set[1], valid_set[1], axis=0)

    # number of data points in train set including the replicated labeled data.
    n = (train_set[0].shape[0])+(n_samples * n_batches)
    # the frequency for the labeled data points to appear in the data set.
    l_freq = n/n_batches
    rng = np.random.RandomState(seed=seed)

    # Create the labeled and unlabeled data evenly distributed across classes.
    x_l, y_l, x_u, y_u = _create_semi_supervised(train_set, n_labeled, rng)

    # Filter out the features with a low standard deviation.
    if filter_std > .0:
        idx_keep = np.std(x_u, axis=0) > filter_std
        x_l, x_u = x_l[:, idx_keep], x_u[:, idx_keep]
        valid_set = (valid_set[0][:, idx_keep], valid_set[1])
        test_set = (test_set[0][:, idx_keep], test_set[1])

    # Interleave labelled and unlabelled datasets
    col_x, col_y = np.zeros((n, x_l.shape[1])), np.zeros((n, y_l.shape[1]))
    i = 0
    u_count = 0
    y_u = np.zeros(y_u.shape)
    while i < n:
        if i % l_freq == 0:
            # add labeled datapoint in order to ensure nicely sized batchsizes.
            indices = np.array(range(n_labeled))
            if not n_samples == n_labeled:
                indices = indices[rng.randint(0, n_labeled, size=n_samples)]
            j = 0
            for idx in indices:
                col_x[i+j] = x_l[idx]
                col_y[i+j] = y_l[idx]
                j += 1
            i += n_samples
        else:
            col_x[i] = x_u[u_count]
            col_y[i] = y_u[u_count]
            u_count += 1
            i += 1
    train_set = (col_x, col_y)
    valid_set = _pad_targets(valid_set)
    test_set = _pad_targets(test_set)

    return train_set, test_set, valid_set