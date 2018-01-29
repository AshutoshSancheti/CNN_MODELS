## y will be 10*1
## Gotta convert the image from 32*32*3 into 3072 i.e. X is 3072*1
## the weights will be 10*3072 and bias 10*1
## y = w*X + b
import numpy as np
import cv2

class LinearClassifier(object):
    ##reg_lambda is the regularisation parameter and the eta is our learning rate
    def __init__(self, n_output, n_input, reg_lambda = 0.0, eta = 0.0, epochs = 500):

        self.n_output = n_output
        self.n_input = n_input
        self.reg_lambda = reg_lambda
        self.eta = eta
        self.epochs = epochs
        self.weights = self.random_initialize_wts()

#just check whether we need to use the cv2.imread for img format
    def image_as_array(self, img):
        w, h, n = im.width, im.height, im.channels
        modes = {1: "L", 3:"RGB", 4: "RGBA"}
        if n not in modes:
            raise StandardError('unsupproted no. of channels:'.format(n))
        out = np.asarray(im) if n == 1 else np.asarray[:,:,::-1]
        return out

## Need to come up with a better way
###################### CHECK THE WORKING OF THE RESHAPE FUNCTION ################
    def random_initialize_wts(self, X, y):
        w = np.random.uniform(-1.0, 1.0, size = (X.shape[0]*y.shape[0]))
        # need to reshape into 10*3072
        w = w.reshape(y.shape[0], X.shape[0])
        return w

    def random_bias(self, y):
        b = np.random.uniform(-.0, 1.0, size = (y.shape[0]))
        b = b.reshape(y.shape[0], 1)
        return b
    
    def forward_feed(self):

    def cost(self):

    def get_gradient(self):

    def regularisation(self):

    def back_prop(self):

    def SVMClassifier(self):

    def fit(self):
        
