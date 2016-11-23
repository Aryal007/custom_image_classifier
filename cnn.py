#Train a CNN to identify photos 

import os
import sys

import numpy as np
from skimage import io
from skimage import transform
from tflearn.data_utils import shuffle

from model import setup_model

SIZE = (32, 32)

image_dir = os.path.abspath("images")		#image_dir = /home/bibek/Downloads/generate_imagedata/imageclassifier/images
circle = io.imread_collection(os.path.join(image_dir, "circle/*"))		#circle = full path to all the files inside circle directory
triangle = io.imread_collection(os.path.join(image_dir, "triangle/*"))	#triangle = full path to all the files inside triangle directory


X_circle = np.asarray([transform.resize(im, SIZE) for im in circle])	
'''
skimage.transform.resize(image, output_shape) => Resize image to match certain size
np.asarray => Input data, in any form that can be converted to an array. This includes lists, lists of tuples, tuples, tuples of tuples, tuples of lists and ndarrays.
transforms the circle images to array 
'''
X_triangle = np.asarray([transform.resize(im, SIZE) for im in triangle])

X = np.concatenate((X_circle, X_triangle), axis=0)
Y = np.concatenate((np.ones(X_circle.shape[0]),
                    np.zeros(X_triangle.shape[0])))

Y = np.zeros((X.shape[0], 2))
Y[:X_circle.shape[0], 1] = np.ones(X_circle.shape[0])
Y[-X_triangle.shape[0]:, 0] = np.ones(X_triangle.shape[0])


n_training = int(X.shape[0] * .66)
X, Y = shuffle(X, Y)
X, X_test, Y, Y_test = X[:n_training], X[n_training:], Y[:n_training], Y[n_training:]

model = setup_model()

# Train it! We'll do 100 training passes and monitor it as it goes.
model.fit(X, Y, n_epoch=1000, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=100, snapshot_epoch=True,
          run_id='classifier')

model.save("classifier.tfl")
