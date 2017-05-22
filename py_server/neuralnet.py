import cPickle
import gzip
import matplotlib.pyplot as plt
import numpy
import operator
import os
import re
from scipy import misc

######################
# Variable Setup
######################
image_size_x = 28
image_size_y = 28
dataset_name = 'mnist.pkl.gz'

#alphas = [0.001,0.01,0.1,1,10,100,1000] #choose the alpha
alphas = [0.1]
training_iteration = 10000 #Total iteration to train
n_hidden = 88 #Number of hidden layer, number between input_layer and output_layer
n_output = 10  #number of output layer, 10 digits
######################

def show_as_image(t):
    t2d = numpy.resize(t, [image_size_x, image_size_y])
    plt.imshow(t2d)
    plt.show()

######################
# Load the dataset
######################
#print "Loading " + dataset_name
#f = gzip.open(dataset_name, 'rb')
#train_set, valid_set, test_set = cPickle.load(f)
#f.close()
#print "Done loading " + dataset_name
regex = r"wnd_\d+_(?P<label>\d+).jpg"

train_img_path = "train_set/"
list_train_img = os.listdir(train_img_path)

dataY = [int(re.match(regex, path).group('label')) for path in list_train_img if re.search(regex,path)]
data = [misc.imread(train_img_path+path, flatten=True) for path in list_train_img if re.search(regex,path)]

data = [misc.imresize(image,(28,28)) for image in data]
data = [image.flatten() for image in data]

dataX = numpy.asarray(data)
dataY = numpy.asarray(dataY)

# print dataX
# print"***************\n"
# print dataY

n = len(dataX)
m = len(dataX[0])

#print n,m
#show first ten image
# for i in range(10):
#     print "Showing number %s" % dataY[i]
#     show_as_image(dataX[i])

#########################
# Training: Neural Network
#########################
def sigmoid(x):
    return (1 + numpy.tanh(x / 2.0)) / 2

def sigmoid_output_to_derivative(output):
    return output * (1-output)

def convert_y(y, m):
    return [[1 if y[j] == i else 0 for i in range(m)] for j in range(len(y))]

y = convert_y(dataY, n_output)

for alpha in alphas:
    print "\nTraining With Alpha:", alpha
    numpy.random.seed(1)
    
    #initialize
    weight0 = 2 * numpy.random.random((m, n_hidden)) - 1
    weight1 = 2 * numpy.random.random((n_hidden, n_output)) - 1
    
    for j in xrange(training_iteration):
        #Forward Propagation
        layer0 = dataX
        layer1 = sigmoid(numpy.dot(layer0, weight0))
        layer2 = sigmoid(numpy.dot(layer1, weight1))

        #Back Propagation
        layer2_error = layer2 - y
        layer2_delta = layer2_error * sigmoid_output_to_derivative(layer2)
        if j % 100 == 0:
            print "Iteration -", str(j), "Training Error:" + str(numpy.mean(numpy.abs(layer2_error)))

        layer1_error = layer2_delta.dot(weight1.T)
        layer1_delta = layer1_error * sigmoid_output_to_derivative(layer1)

        #update the weight
        weight1 -= alpha * layer1.T.dot(layer2_delta)
        weight0 -= alpha * layer0.T.dot(layer1_delta)

print "\nEND"