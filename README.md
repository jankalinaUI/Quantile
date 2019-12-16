# An Interquantile Robust Training of Neural Networks, version 1.0

The code in Python performs a robust training of neural networks based on nonlinear 
quantiles, which are also trained by means of neural networks. This is a unique
alternative way of training multilayer perceptrons or radial basis function networks.

Feel free to use or modify the code.

## Requirements

You need to install TensorFlow, Keras, SciPy, NumPy, scikit-learn.

## Usage

* The files have to be run exactly in this order, starting with reading a particular dataset and auxiliary files, performing the training, and presenting the results:
RBFLayer.py, Datasets.py, Evaluation.py, Losses.py, QuantileNetworkClass.py, NetworkTraining.py.

## Authors
  * Tomáš Jurica, The Czech Academy of Sciences, Institute of Computer Science
  * Petra Vidnerová, The Czech Academy of Sciences, Institute of Computer Science
  * Jan Kalina, The Czech Academy of Sciences, Institute of Computer Science
 
## Contact

Do not hesitate to contact us (petra@cs.cas.cz) or write an Issue.

## How to cite

Please consider citing the following:

Kalina J, Vidnerová P (2020): An interquantile approach to robust training of neural networks. Submitted.

## Acknowledgement

This work was supported by projects 19-05704S and TN01111124.