__author__ = 'larsma'

import theano.sandbox.cuda # TODO delete
theano.sandbox.cuda.use('gpu3') # TODO delete
import theano
from training.train import TrainModel
from lasagne_extensions.nonlinearities import rectify
from data_preparation import mnist
from models import ADGMSSL

def run_adgmssl_mnist():
    """
    Train a auxiliary deep generative model on the mnist dataset with 100 evenly distributed labels.
    """
    n_labeled = 100 # The total number of labeled data points.
    n_samples = 100 # The number of sampled labeled data points for each batch.
    n_batches = 600 # The number of batches.
    mnist_data = mnist.load_semi_supervised(n_batches=n_batches, n_labeled=n_labeled, n_samples=n_samples,
                                            filter_std=0.1, train_valid_combine=True)

    n, n_x = mnist_data[0][0].shape # Datapoints in the dataset, input features.
    bs = n / n_batches # The batchsize.

    # Initialize the auxiliary deep generative model.
    model = ADGMSSL(n_x=n_x, n_a=100, n_z=100, n_y=10, a_hidden=[500, 500],
                    z_hidden=[500, 500], xhat_hidden=[500, 500], y_hidden=[500, 500],
                    trans_func=rectify, x_dist='bernoulli')

    # Get the training functions.
    f_train, f_test, f_validate, train_args, test_args, validate_args = model.build_model(*mnist_data)
    # Update the default function arguments.
    train_args['inputs']['batchsize'] = bs
    train_args['inputs']['batchsize_labeled'] = n_samples
    train_args['inputs']['beta'] = 0.01*n
    train_args['inputs']['learningrate'] = 3e-4
    train_args['inputs']['beta1'] = 0.9
    train_args['inputs']['beta2'] = 0.999
    train_args['inputs']['samples'] = 10
    test_args['inputs']['samples'] = 100 # sample 100 times to get estimation of test error.
    validate_args['inputs']['samples'] = 1

    # Define training loop.
    evaluate_every_n_epoch = 5
    train = TrainModel(model=model, anneal_lr=.75, anneal_lr_freq=200, output_freq=evaluate_every_n_epoch)
    train.add_initial_training_notes("Training the auxiliary deep generative model with %i labels." % n_labeled)
    train.train_model(f_train, train_args,
                      f_test, test_args,
                      f_validate, validate_args,
                      n_train_batches=n_batches,
                      n_epochs=3000)

if __name__ == "__main__":
    run_adgmssl_mnist()
