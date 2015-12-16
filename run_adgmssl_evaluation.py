__author__ = 'larsmaaloe'

import theano
from lasagne_extensions.nonlinearities import rectify
from data_preparation import mnist
from models import ADGMSSL
from lasagne_extensions.layers import MergeLayer, Layer, get_output
from utils import env_paths
import theano.tensor as T
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def run_adgmssl_mnist():
    """
    Evaluate a auxiliary deep generative model on the mnist dataset with 100 evenly distributed labels.
    """

    # Load the mnist supervised dataset for evaluation.
    (train_x, train_t), (test_x, test_t), (valid_x, valid_t) = mnist.load_supervised(filter_std=0.0,
                                                                                     train_valid_combine=True)

    # Initialize the auxiliary deep generative model.
    model = ADGMSSL(n_x=train_x.shape[-1], n_a=100, n_z=100, n_y=10, a_hidden=[500, 500],
                    z_hidden=[500, 500], xhat_hidden=[500, 500], y_hidden=[500, 500],
                    trans_func=rectify, x_dist='bernoulli')

    model.load_model(20151209002003)  # Load trained model. See configurations in the log file.

    # Evaluate the test error of the ADGM.
    mean_evals = model.get_output(test_x, 100)  # 100 MC to get a good estimate for the auxiliary unit.
    t_class = np.argmax(test_t, axis=1)
    y_class = np.argmax(mean_evals, axis=1)
    class_err = np.sum(y_class != t_class) / 100.
    print "test set 100-samples: %0.2f%%." % class_err

    # Evaluate the active units in the auxiliary and latent distribution.
    f_a_mu_logvar = theano.function([model.sym_x_l], get_output([model.l_a_mu, model.l_a_logvar], model.sym_x_l))
    q_a_mu, q_a_logvar = f_a_mu_logvar(test_x)
    log_pa = -0.5 * (np.log(2 * np.pi) + (q_a_mu ** 2 + np.exp(q_a_logvar)))
    log_qa_x = -0.5 * (np.log(2 * np.pi) + 1 + q_a_logvar)
    diff_pa_qa_x = (log_pa - log_qa_x).mean(axis=(1, 2))
    mean_diff_pa_qa_x = np.abs(np.mean(diff_pa_qa_x, axis=0))

    inputs = {model.l_x_in: model.sym_x_l, model.l_y_in: model.sym_t_l}
    f_z_mu_logvar = theano.function([model.sym_x_l, model.sym_t_l],
                                    get_output([model.l_z_mu, model.l_z_logvar], inputs))
    q_z_mu, q_z_logvar = f_z_mu_logvar(test_x, test_t)
    log_pz = -0.5 * (np.log(2 * np.pi) + (q_z_mu ** 2 + np.exp(q_z_logvar)))
    log_qz_x = -0.5 * (np.log(2 * np.pi) + 1 + q_z_logvar)
    diff_pz_qz_x = (log_pz - log_qz_x).mean(axis=(1, 2))
    mean_diff_pz_qz_x = np.abs(np.mean(diff_pz_qz_x, axis=0))

    plt.figure()
    plt.subplot(111, axisbg='white')
    plt.plot(sorted(mean_diff_pa_qa_x)[::-1], color="#c0392b", label=r"$\log \frac{p(a_i)}{q(a_i|x)}$")
    plt.plot(sorted(mean_diff_pz_qz_x)[::-1], color="#9b59b6", label=r"$\log \frac{p(z_i)}{q(z_i|x)}$")
    plt.grid(color='0.9', linestyle='dashed', axis="y")
    plt.xlabel("stochastic units")
    plt.ylabel(r"$\log \frac{p(\cdot)}{q(\cdot)}$")
    plt.ylim((0, 2.7))
    plt.legend()
    plt.savefig("output/diff.png", format="png")

    # Sample 100 random normal distributed samples with fixed class y in the latent space and generate xhat.
    table_size = 10
    samples = 1
    z = np.random.random_sample((table_size ** 2, 100))
    y = np.eye(10, k=0).reshape(10, 1, 10).repeat(10, axis=1).reshape((-1, 10))
    xhat = model.f_xhat(z, y, samples)

    plt.figure(figsize=(20, 20), dpi=300)
    i = 0
    img_out = np.zeros((28 * table_size, 28 * table_size))
    for x in range(table_size):
        for y in range(table_size):
            xa, xb = x * 28, (x + 1) * 28
            ya, yb = y * 28, (y + 1) * 28
            im = np.reshape(xhat[i], (28, 28))
            img_out[xa:xb, ya:yb] = im
            i += 1
    plt.matshow(img_out, cmap=plt.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.savefig("output/mnist.png", format="png")


if __name__ == "__main__":
    run_adgmssl_mnist()
