 
"""This is a lower level implementation of MNIST in CNN
   The high level implementation using tf.layers is in cnn_mnist.py
 """
 ####ADDED   
        ##LEARNING RATE DECAY
        ##A THIRD CONVOLUTION LAYER
        ##NORMALIZED DATA BETWEEN 0 AND 1
 
from __future__ import print_function
import tensorflow as tf
import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import MinMaxScaler
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

scaler = MinMaxScaler(feature_range = (0, 1))
#rescaledX = scaler.fit_transform(X)
test_data = 'MNIST_csv/test.csv' ##Kaggle data

req_cols_test = np.zeros((784), np.int32)
for i in range(784):
    req_cols_test[i] = i
    
xtest = pd.read_csv(test_data, usecols = req_cols_test, skiprows = [0], header = None, dtype = np.float32)
x_test = np.asarray(xtest, dtype = np.float32)
x_test = x_test
scaled_x_test = scaler.fit_transform(x_test)

##This part only to pass a part of test data i.e., first 100 test images
#x_test1 = np.zeros((100,784), np.float32)
#x_test1 = x_test1 + x_test[0:100,0:784]

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape = [None, 784])
y_ = tf.placeholder(tf.float32, shape = [None, 10])


#utility functions
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)
    
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)
    
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')
    
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')  #kernel size is 2x2 and stride of 2
    
    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    #P1 = tf.nn.max_pool(A, ksize = [1,8,8,1], strides = [1,8,8,1], padding = 'SAME')

x_image = tf.reshape(x, [-1, 28, 28, 1])

#1st layer	
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#2nd layer
W_conv2 = weight_variable([5,5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

#3rd conv layerr before the second max_pool layer
W_conv3 = weight_variable([5, 5, 64, 64])
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)

h_pool2 = max_pool_2x2(h_conv3)

#Dense connected layer
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#LEARNING RATE
#global_step = tf.Variable(0, trainable = False)
#starter_learning_rate = 0.0005
#learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 8000, 0.0005, staircase=True)

learning_rate = tf.constant(0.0001)

#Optimization
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def test_func(x_test, a, b, c, d, e, f, g, h, i, j):
    tx_flat_test = tf.placeholder(tf.float32, shape = [None, 784])
    tx_flat_test = x_test
    print (x_test.shape)
    tx_test_in = tf.reshape(tx_flat_test, [-1,28,28,1])
    #1st conv and pool
    conv1 = conv2d(tx_test_in, a)
    relu1 = tf.nn.relu(conv1 + b)
    max1 = max_pool_2x2(relu1)
    #2nd conv and pool
    conv2 = conv2d(max1, c)
    relu2 = tf.nn.relu(conv2 + d)
    
    conv3 = conv2d(relu2, i)
    relu3 = tf.nn.relu(conv3 + j)
    max2 = max_pool_2x2(relu3)    
    #1st FC layer
    flat = tf.reshape(max2, [-1,7*7*64]) #shape is [1, 7*7*64]
    fc = tf.nn.relu(tf.matmul(flat, e) + f)   #output is [1, 1024]    
    #FC1 layer to y_logit or FC2
    y_logit = tf.matmul(fc, g) + h  #output is [1,10]   
    #y_logit to softmax 
    softmax = tf.nn.softmax(y_logit)
    return softmax

#Reduce the size of test_data passed, create mini batch function
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(8000):
        batch = mnist.train.next_batch(50)
        
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: scaler.fit_transform(batch[0]), y_: batch[1], keep_prob: 1.0})
            #train_accuracy = accuracy.eval(feed_dict={x: (batch[0]), y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
            learning_rate = learning_rate/(1 + 0.005)
            print(sess.run(learning_rate))
    
        train_step.run(feed_dict = {x: scaler.fit_transform(batch[0]), y_: batch[1], keep_prob: 0.5})
        #train_step.run(feed_dict={x: (batch[0]), y_: batch[1], keep_prob: 0.5})
        #learning_rate = learning_rate/(1 + 0.000000006)
        
    #print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    my_file = open("Results8.txt", "w")
    #my_file.write("ImageId,Labels\n")
    for i in range(27472,28000):
        #x_test1 = x_test[i,0:784]
        x_test1 = scaled_x_test[i,0:784]
        test_logits = test_func(x_test1, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2, W_conv3, b_conv3)
        max_index = np.argmax(test_logits.eval(), axis = 1) #its necessary to add .eval()
        max_index_scalar = np.asscalar(max_index)
        my_file.write(str(i+1) + "," + str(max_index_scalar) + "\n")
    
    my_file.close()
    #print (test_logits.eval()) #Thsi fumctionwas used to print the one-hot outputs of the x_test1


"""	
line 142, 138, 150
#Trying a second way to encode using keras
##from tf.keras.utils import to_categorical
##onehot_y_train = tf.keras.utils.to_categorical(y_train)

onehot_y_train = np.eye(10)[y_train.reshape(-1)]  # here 10 is the number of classes
#reshape(-1) makes sure that the array is of the form [1, 2, 3, .....]

print (onehot_y_train)
print (onehot_y_train.shape)
"""

"""
#Reduce the size of test_data passed, create mini batch function
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(200):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    #print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    test_logits = test_func(x_test1, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2)
    
    print (test_logits.eval())
"""	
