#Train a CNN to identify photos 

import os
import sys

import numpy as np
from skimage import io
from skimage import transform
from tflearn.data_utils import shuffle

from model import setup_model

SIZE = (2, 2)

np.set_printoptions(threshold=np.nan)		#to view the full printed array

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
'''
skimage.transform.resize(image, output_shape) => Resize image to match certain size
np.asarray => Input data, in any form that can be converted to an array. This includes lists, lists of tuples, tuples, tuples of tuples, tuples of lists and ndarrays.
transforms the triangle images to array 
'''
X = np.concatenate((X_circle, X_triangle), axis=0)
'''
Concatenates the input of both the arrays into 1 as in example below
>>> a = np.array([[1, 2], [3, 4]])
>>> b = np.array([[5, 6]])
>>> np.concatenate((a, b), axis=0)
array([[1, 2],
       [3, 4],
       [5, 6]])
'''
# print "Circle is"
# print X_circle
# print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
# print "Triangle is"
# print X_triangle
# print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
# print "Concatenation is"
# print X
# print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
# print "Shape (total number) of circle is: ",X_circle.shape[0]
# print "Shape (total number) of triangle is: ",X_triangle.shape[0]
# print "Np.ones X_circle.shape[0]: ",np.ones(X_circle.shape[0])
# print "Np.ones X_triangle.shape[0]: ",np.zeros(X_circle.shape[0])
Y = np.concatenate((np.ones(X_circle.shape[0]),
                    np.zeros(X_triangle.shape[0])))
'''
Shape gives the number of circles and traingles in our input data.
If there are 500 circles and 600 triangles in the set,
X_circle.shape[0] = 500
X_triangle.shape[0] = 600

>>> s = 3
>>> np.ones(s)
array([ 1.,  1.,  1.])

>>> s = 3
>>> np.zeros(s)
array([ 0.,  0.,  0.])

np.concatenate((np.ones(X_circle.shape[0]), np.zeros(X_triangle.shape[0]))) gives a one dimensional array
with first number of circle of elements as one and rest elements as zeros  
'''
Y = np.zeros((X.shape[0], 2))
'''
X.shape[0] gives total number of data. For example, if there  are 500 circles and 600 triangles, X.shape[0]
gives 500+600 = 1100 as the concatenated length of X-circle and X_triangle

np.zeros((6,2)) gives a 6 X 2 dimensional array with all elements as 0
Eg:
[[ 0.  0.]
 [ 0.  0.]
 [ 0.  0.]
 [ 0.  0.]
 [ 0.  0.]
 [ 0.  0.]]
'''
# print "X_shape: ",X.shape[0]
# print np.zeros((6,2))

Y[:X_circle.shape[0], 1] = np.ones(X_circle.shape[0])
# print Y
# [[ 0.  1.]
#  [ 0.  1.]
#  [ 0.  1.]
#  [ 0.  0.]
#  [ 0.  0.]
#  [ 0.  0.]]
Y[-X_triangle.shape[0]:, 0] = np.ones(X_triangle.shape[0])
# print Y
# [[ 0.  1.]
#  [ 0.  1.]
#  [ 0.  1.]
#  [ 0.  0.]
#  [ 0.  0.]
#  [ 0.  0.]]
'''
What we are doing here is generating a shape X 2 dimensional array for our expected output
For example, we have 3 circles and 4 triangles, total number = 7
First we create a 7 X 2 dimensional array of all zeros (since we have two output class 0 and 1)
We then put the value of first circle.shape[0](i.e. 3) elements as [0. 1.] to denote a circle
Then we put the value of last triangle.shape[0](i.e. 4) elements as [1., 0.] to denote a triangle
'''
n_training = int(X.shape[0] * .66)
# print n_training
# 6*0.66 = 3.96 = integer 3
# print X
# print Y
X, Y = shuffle(X, Y)
print X,Y
exit(0)
X, X_test, Y, Y_test = X[:n_training], X[n_training:], Y[:n_training], Y[n_training:]

model = setup_model()

# Train it! We'll do 100 training passes and monitor it as it goes.
model.fit(X, Y, n_epoch=1000, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=100, snapshot_epoch=True,
          run_id='classifier')

model.save("classifier.tfl")
