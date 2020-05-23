# Next Centroid Classifier for MNIST

A naive, experimental next-centroid-classifier for the MNIST dataset.

The path to the folder with the MNIST dataset can be set as an parameter.
In order to import the MNIST dataset the external module 
[mnist](https://pypi.python.org/pypi/python-mnist/) is required.

#### Feature extraction:

- **dumb:**  
  A simple feature vector consisting of two image-rows and one image-column. Accuracy ~70%.

- **pca:**  
  Projects the images to a 50 dimensional feature vector. Accuracy ~85%.

#### Distance measure:

- euclidean norm.

Additional experiments with alternative feature types (Harris, BRIEF) have been conducted,
but their binary feature vectors are unsuited for metrical distance measures.
