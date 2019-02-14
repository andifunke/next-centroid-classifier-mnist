# Next Centroid Classifier for MNIST

A naive next-centroid-classifier for the MNIST dataset as an experiment.

The path to the folder with the MNIST-dataset can be set as an parameter.
To import MNIST data the external module [mnist](https://pypi.python.org/pypi/python-mnist/) is needed.

feature extraction:

dumb:
A quite simple feature vector consisting of two image-rows and one image-column. Accuracy around 70%.

pca:
Projects the images to a 50 dimensional feature vector. Accuracy around 85%.

Distance measure: euclidean norm.

I have also conducted experiments with additional feature types (Harris, BRIEF), but their binary feature vectors are unsuited for metric distance measures.
