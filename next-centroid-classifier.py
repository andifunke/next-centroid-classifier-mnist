# -*- coding: utf-8 -*-
import numpy as np
import cv2
from sys import argv
from mnist import MNIST
from math import sqrt
from sklearn.decomposition import PCA
from time import time
from skimage.feature import corner_harris, corner_peaks, BRIEF


h, w = 28, 28
path = './MNIST'
if len(argv) > 1:
    path = argv[1]


# calculate PCA
# source: http://scikit-learn.org/stable/auto_examples/applications/face_recognition.html
def get_pca(train, test):
    n_components = 50
    print("Extracting the top %d eigenfaces" % n_components)
    t0 = time()
    pca = PCA(n_components=n_components, svd_solver='randomized',
              whiten=True).fit(train)
    print("done in %0.3fs" % (time() - t0))
    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    train_pca_result = pca.transform(train)
    test_pca_result = pca.transform(test)
    print("done in %0.3fs" % (time() - t0))
    return train_pca_result, test_pca_result


def dumb(img):
    horizontal = np.append(img[9], img[18])
    vertical = []
    for j in range(len(img[14])):
        vertical.append(img[14][j])
    return tuple(np.append(horizontal, vertical))


# not working correctly
def harris(img):
    img2 = np.float32(img)
    dst = cv2.cornerHarris(img2, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    return dst


# not working correctly
def brief(img):
    keypoints = corner_peaks(corner_harris(img), min_distance=1)
    extractor = BRIEF(descriptor_size=64, patch_size=12)
    extractor.extract(img, keypoints)
    print extractor.descriptors
    return extractor.descriptors


def get_features(img, func):
    if func is None:
        return img
    return func(img)


def convert_mnistraw_to_nparray(mnistraw):
    nparrays = []
    for entry in mnistraw:
        nparrays.append(np.reshape(np.asarray(entry), (h, w)))
    return nparrays


def dist(p, q):
    # print p, q
    sum = 0
    for i, val in enumerate(p):
        sum += (val - q[i])**2
    return sqrt(sum)


def process(train_data, test_data, func, name):
    # dict of feature vectors for all 10 classes
    train_classes = {key: [] for key in range(10)}
    # print classes

    # assign feature vectors of train data to classes
    for index, label in enumerate(train_labels):
        # print index, label
        features = get_features(train_data[index], func)
        train_classes[label].append(features)

    # determine centroids
    centroids = {}
    for clazz, feature_vectors in train_classes.items():
        feature_size = len(feature_vectors[0])
        count = float(len(feature_vectors))
        centroid = [0 for i in range(feature_size)]
        for vector in feature_vectors:
            for feature, value in enumerate(vector):
                centroid[feature] += value
        for i in range(feature_size):
            centroid[i] /= count
        centroids[clazz] = tuple(centroid)
    # print centroids

    # assign indices of test data to centroids
    test_labelassignments = []
    for index, image in enumerate(test_data):
        vector = get_features(image, func)
        distances = {}
        for clazz, centroid in centroids.items():
            distances[dist(centroid, vector)] = clazz
        # print distances
        # print sorted(distances)
        min_dist = min(distances)
        # print min_dist
        nearest_centroid = distances[min_dist]
        # print nearest_centroid
        # test_classes[nearest_centroid] = index
        test_labelassignments.append(nearest_centroid)

    # print test_classes
    print test_labelassignments
    print test_labels

    right = 0
    for index, label in enumerate(test_labels):
        if label == test_labelassignments[index]:
            right += 1

    precision = float(right) / len(test_labels)
    print name+':', 'precision =', right, '/', len(test_labels), '=', precision


# load raw mnist data
mndata = MNIST(path)
mndata.load_training()
mndata.load_testing()

# convert raw mnist images to numpy arrays
train_np1D = np.asarray(mndata.train_images)
train_np2D = convert_mnistraw_to_nparray(mndata.train_images)
train_labels = mndata.train_labels
test_np1D = np.asarray(mndata.test_images)
test_np2D = convert_mnistraw_to_nparray(mndata.test_images)
test_labels = mndata.test_labels

# process different feature sets
process(train_np2D, test_np2D, dumb, 'dumb')
train_pca, test_pca = get_pca(train_np1D, test_np1D)
process(train_pca, test_pca, None, 'pca')
# process(train_np2D, test_np2D, harris, 'harris')
# process(train_np2D, test_np2D, brief, 'BRIEF')
