Auxiliary Deep Generatives Models
=======
This repository is the implementation of the auxiliary deep generative model presented at the workshop on
`advances in approximate Bayesian inference <http://approximateinference.org>`_, NIPS 2015. The
`article <http://approximateinference.org/accepted/MaaloeEtAl2015.pdf>`_ show state-of-the-art performance on MNIST and
will be submitted to ICML 2016 in an extended format where more datasets are included.


The implementation is build on the `Parmesan <https://github.com/casperkaae/parmesan>`_, `Lasagne <http://github.com/Lasagne/Lasagne>`_ and `Theano <https://github.com/Theano/Theano>`_ libraries.


Installation
------------
Please make sure you have installed the requirements before executing the python scripts.


**Install**


.. code-block:: bash

  git clone https://github.com/casperkaae/parmesan.git
  cd parmesan
  python setup.py develop



Examples
-------------
The repository primarily includes


* script running a new model on the MNIST datasets with only 100 labels - *run_adgmssl_mnist.py*.
* script evaluating a trained model (see model specifics in *output/.) - *run_adgmssl_evaluation.py*.
* iPython notebook where all training is implemented in a single scipt - *run_adgmssl_mnist_notebook.ipynb*.


Please see the source code and code examples for further details.


Comparison of the training convergence between the adgmssl, adgmssl with deterministic auxiliary units and the dgmssl.


.. image:: /output/train.png


Showing the information contribution from the auxiliary and the latent units a and z respecively.


.. image:: /output/diff.png


A random sample from the latent space run through the generative model.


.. image:: /output/mnist.png