import os
import sys
import numpy as np
import tensorflow as tf
import scipy as sc
import scipy.io
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
 
#from keras.models import Model
#from keras.models import Sequential
#from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
#from keras.layers.pooling import MaxPooling2Ds
#from keras.layers.normalization import BatchNormalization

class CONFIG:
    IMAGE_WIDTH = 300
    IMAGE_HEIGHT = 225
    COLOR_CHANNELS = 3
    NOISE_RATIO = 0.6
    MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) 
    VGG_MODEL = 'pretrained-model/imagenet-vgg-verydeep-19.mat'

#Images for report
report_images = []
learning_rate = 2.0

#path to the pre-trained mpdels
PATH_VGG19 = "C:\pretrained_model\imagenet-vgg-verydeep-19.mat"
img_h = 225 #300
img_w = 300 #400
img_c = 3

noise_ratio = 0.6

#NST cost parameters
alpha = 10
beta = 40
VGG_MEAN = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))  #Availed from the VGG paper for the ImageNet dataset

content_image_path = "persian_cat_content.jpg"
style_image_path = "stone_style.jpg"
content_image_name = "per_cat"
style_image_name = "_stone"


#content_image = cv2.imread(content_image_path)
#content_image = np.reshape(content_image, (1, img_h, img_w, img_c))
#style_image = cv2.imread(style_image_path)
#style_image = np.reshape(style_image, (1, img_h, img_w, img_c))
##path_generated_image = "C:\Users\ashus\NeuralNetworks\Neural_Artistic_Style_Transfer\output\generated_image.jpg"
#path_generated_image = "_stone_style_generated_image.jpg"
style_layers = [('conv1_1', 0.2), ('conv2_1', 0.2), ('conv3_1', 0.2), ('conv4_1', 0.2), ('conv5_1', 0.2)]

def reshape_and_normalize(image):
    image = np.reshape(image, ((1,) + image.shape))
    image = image - CONFIG.MEANS
    return image
    
 
#Select images for generate art
content_image_orig = scipy.misc.imread(content_image_path)
report_images.append(content_image_orig)
content_image = reshape_and_normalize(content_image_orig)

style_image_orig = scipy.misc.imread(style_image_path)
report_images.append(style_image_orig)
style_image = reshape_and_normalize(style_image_orig)

path_generated_image = content_image_name + '_' + style_image_name + '.png'
    
def vgg19_model_tf(PATH):
    """
        0 is conv1_1 (3, 3, 3, 64)
        1 is relu
        2 is conv1_2 (3, 3, 64, 64)
        3 is relu   
        4 is maxpool
        5 is conv2_1 (3, 3, 64, 128)
        6 is relu
        7 is conv2_2 (3, 3, 128, 128)
        8 is relu
        9 is maxpool
        10 is conv3_1 (3, 3, 128, 256)
        11 is relu
        12 is conv3_2 (3, 3, 256, 256)
        13 is relu
        14 is conv3_3 (3, 3, 256, 256)
        15 is relu
        16 is conv3_4 (3, 3, 256, 256)
        17 is relu
        18 is maxpool
        19 is conv4_1 (3, 3, 256, 512)
        20 is relu
        21 is conv4_2 (3, 3, 512, 512)
        22 is relu
        23 is conv4_3 (3, 3, 512, 512)
        24 is relu
        25 is conv4_4 (3, 3, 512, 512)
        26 is relu
        27 is maxpool
        28 is conv5_1 (3, 3, 512, 512)
        29 is relu
        30 is conv5_2 (3, 3, 512, 512)
        31 is relu
        32 is conv5_3 (3, 3, 512, 512)
        33 is relu
        34 is conv5_4 (3, 3, 512, 512)
        35 is relu
        36 is maxpool
        37 is fullyconnected (7, 7, 512, 4096)
        38 is relu
        39 is fullyconnected (1, 1, 4096, 4096)
        40 is relu
        41 is fullyconnected (1, 1, 4096, 1000)
        42 is softmax
        
    wts = scipy.io.loadmat(PATH)
    wts_layers = wts['layers']
    
    def weights(layer, expected_layer_name):
        wb = wts_layers[0][layer][0][0][2]
        W = wb[0][0]
        b = wb[0][1]
        layer_name = wts_layers[0][layer][0][0][0][0]
        assert layer_name == expected_layer_name
        return W, b

        return W, b

    def _conv2d_relu(prev_layer, layer, layer_name):
        W, b = weights(layer, layer_name)
        W = tf.constant(W)
        b = tf.constant(np.reshape(b, (b.size)))
        conv = tf.nn.conv2d(prev_layer, filter = W, strides = [1,1,1,1], padding = 'SAME') + b
        activate = tf.nn.relu(conv)
        return conv

    def _avgpool(prev_layer):
        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Constructs the graph model.
    graph = {}
    graph['input']   = tf.Variable(np.zeros((1, img_h, img_w, img_c)), dtype = 'float32')
    graph['conv1_1']  = _conv2d_relu(graph['input'], 0, 'conv1_1')
    graph['conv1_2']  = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = _avgpool(graph['conv1_2'])
    graph['conv2_1']  = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2']  = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = _avgpool(graph['conv2_2'])
    graph['conv3_1']  = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2']  = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3']  = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4']  = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = _avgpool(graph['conv3_4'])
    graph['conv4_1']  = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2']  = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3']  = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4']  = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = _avgpool(graph['conv4_4'])
    graph['conv5_1']  = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2']  = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3']  = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4']  = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = _avgpool(graph['conv5_4'])
    """
    vgg = scipy.io.loadmat(PATH)

    vgg_layers = vgg['layers']
    
    def _weights(layer, expected_layer_name):
        wb = vgg_layers[0][layer][0][0][2]
        W = wb[0][0]
        b = wb[0][1]
        layer_name = vgg_layers[0][layer][0][0][0][0]
        assert layer_name == expected_layer_name
        return W, b

        return W, b

    def _relu(conv2d_layer):
        return tf.nn.relu(conv2d_layer)

    def _conv2d(prev_layer, layer, layer_name):
        W, b = _weights(layer, layer_name)
        W = tf.constant(W)
        b = tf.constant(np.reshape(b, (b.size)))
        return tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b

    def _conv2d_relu(prev_layer, layer, layer_name):
        return _relu(_conv2d(prev_layer, layer, layer_name))

    def _avgpool(prev_layer):
        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Constructs the graph model.
    graph = {}
    graph['input']   = tf.Variable(np.zeros((1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.COLOR_CHANNELS)), dtype = 'float32')
    graph['conv1_1']  = _conv2d_relu(graph['input'], 0, 'conv1_1')
    graph['conv1_2']  = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = _avgpool(graph['conv1_2'])
    graph['conv2_1']  = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2']  = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = _avgpool(graph['conv2_2'])
    graph['conv3_1']  = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2']  = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3']  = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4']  = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = _avgpool(graph['conv3_4'])
    graph['conv4_1']  = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2']  = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3']  = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4']  = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = _avgpool(graph['conv4_4'])
    graph['conv5_1']  = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2']  = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3']  = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4']  = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = _avgpool(graph['conv5_4'])
    
    return graph
        
def generate_noise_image(img_h, img_w, img_c, content_image, noise_ratio = 0.5):
    noise_image = np.random.uniform(-20, 20, (1, img_h, img_w, img_c)).astype('float32')
    input_image = noise_image * noise_ratio + content_image * (1.0 - noise_ratio)
    #input_image = np.asarray(noise_image) * noise_ratio + np.asarray(content_image) * (1.0 - noise_ratio)
    return input_image

    
def save_image(path, image):
    image = image + CONFIG.MEANS
    image = np.clip(image[0], 0, 255).astype('uint8')
    #cv2.imwrite(path, image)
    scipy.misc.imsave(path, image)
        
def compute_content_cost(content_img, gene_img):
    #content_mg - tensor of dimension (1, n_H, n_W, n_C), hidden layer activations  
    #gene_img - tensor of dimension (1, n_H, n_W, n_C), hidden layer activations 
    a, height, width, channels = gene_img.get_shape().as_list()
    re_content_img = tf.transpose(tf.reshape(content_img, [(height * width), channels]))
    re_gene_img = tf.transpose(tf.reshape(gene_img, [(height * width), channels]))
    cost_content = (1/(4 * channels * height * width)) * tf.reduce_sum(tf.square(tf.subtract(re_content_img,re_gene_img)))
    return cost_content
    
def gram_matrix(A):
    #A - matrix of shape (channels, height * width)
    #Gram matrix of A, of shape (channels, channels)
    gram_mat = tf.matmul(A, tf.transpose(A))
    return gram_mat

def compute_layer_style_cost(style_img, gene_img):
    #style_img - tensor of dimension (1, height, width, channels), hidden layer activations 
    #gene_img - tensor of dimension (1, height, width, channels), hidden layer activations 
    
    a, height, width, channels = gene_img.get_shape().as_list()
    style_img= tf.transpose(tf.reshape(style_img, [(height * width), channels]))
    gene_img = tf.transpose(tf.reshape(gene_img, [(height * width), channels]))
    gram_style = gram_matrix(style_img)
    gram_gene = gram_matrix(gene_img)
    cost_style_layer = (1 / (4 * channels * channels *(height * width)*(height * width)))*(tf.reduce_sum(tf.square(tf.subtract(gram_style,gram_gene))))     
    return cost_style_layer

style_layers = [('conv1_1', 0.2), ('conv2_1', 0.2), ('conv3_1', 0.2), ('conv4_1', 0.2), ('conv5_1', 0.2)]

sess = tf.InteractiveSession()
def compute_style_cost(style_layers, model, sess):
    cost_style = 0
    for layer_name, const in style_layers:
        layer_out = model[layer_name]
        style_activ = sess.run(layer_out)
        gene_activ = layer_out
        cost_style_layer = compute_layer_style_cost(style_activ, gene_activ)
        cost_style += (const * cost_style_layer)
    return cost_style
    
def total_cost(cost_content, cost_style, alpha = 10, beta = 40):
    net_cost = (alpha * cost_content) + (beta * cost_style)
    return net_cost

# Reset the graph
# Start interactive session
# Initialize a noisy image by adding random noise to the content_image
# load VGG19 model
# Assign the content image to be the input of the VGG model.

# Select the output tensor of layer conv4_2
# Set a_C to be the hidden layer activation from the layer we have selected
# Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2'] 
# and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
# when we run the session, this will be the activations drawn from the appropriate layer, with G as input.

# Compute the content cost
# Assign the input of the model to be the "style" image
# Compute the style cos
# Get the
# define optimizer - using adam(1 line)
# define train_step (1 line)

tf.reset_default_graph()
sess = tf.InteractiveSession()
noise_img = generate_noise_image(img_h, img_w, img_c, content_image, noise_ratio);
model = vgg19_model_tf("C:\pretrained_model\imagenet-vgg-verydeep-19.mat")
sess.run(model['input'].assign(content_image))

out = model['conv4_2']
a_C = sess.run(out)
a_G = out
# a_G references model['conv4_2'] and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
# when we run the session, this will be the activations drawn from the appropriate layer, with G as input.

content_cost = compute_content_cost(a_C, a_G)
sess.run(model['input'].assign(style_image))
style_cost = compute_style_cost(style_layers, model, sess)
net_cost = total_cost(content_cost, style_cost, alpha, beta)
optimizer = tf.train.AdamOptimizer(2.0)
train = optimizer.minimize(net_cost)

def model_nn(sess, input_image, path_generated_image, iterations = 200):
    sess.run(tf.global_variables_initializer())
    sess.run(model['input'].assign(input_image))

    for i in range(iterations):
        sess.run(train)
        generated_img = sess.run(model['input'])
        if (i%20 == 0):
            J_c, J_s, J_t = sess.run([content_cost, style_cost, net_cost])
            print ("Iteration no. " + str(i) + ": ")
            print ("Content cost = " + str(J_c))
            print (", Style cost = " + str(J_s))
            print (", Total cost = " + str(J_t))
            save_image("output/" + str(i) + ".png", generated_img)
            
    save_image(path_generated_image, generated_img)
    return generated_img
    
model_nn(sess, noise_img, path_generated_image, iterations = 200)
read_generated = scipy.misc.imread(path_generated_image)
report_images.append(read_generated)