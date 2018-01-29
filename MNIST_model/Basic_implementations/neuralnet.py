import numpy as np
from scipy.special import expit
import sys

#A neural network with 1 hidden layer

class NeuralNetMLP(object):
    def __init__(self, n_output, n_input, n_hidden = 30, l1 = 0.0, l2 = 0.0, epochs = 500, eta = 0.0001, alpha = 0.0, decrease_const = 0.0, shuffle = True, minibatches =1, random_state = None):

        np.random.seed(random_state)
        self.n_output = n_output
        self.n_input - n_input
        self.n_hidden = n_hidden
        self.w1, self.w2 = self._initialize_weights()
        self.l1 = l1
        self.l2 = l2
        #l1 and l2 are the type of regularisation used
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.decrease_count = decrease_count
        self.shuffle = shuffle
        self.minibatches = minibatches

    def _encode_labels(self, y , k):
        onehot = np.zeros(k, y.shape[0])
        for idx, val in enumerate(y):
            onehot[val, idx] = 1.0
        return onehot

    def _initialize_weights(self):
        #w1 is the weights between 1st and hidden layer
        w1 = np.random.uniform(-1.0, 1.0, size = self.n_hidden*(self.n_features + 1))
        #Here the w1 with no. of elements = n_hidden*(n_features +1) where +1 for the bias perceptron
        #the above function randomly initializes the elements of w1 from float values between -1 and 1. it's necessary to mention 1.0 and not just 1
        w1 = w1.reshape(self.n_hidden, self.features+1)
        #to convert w1 into a 2-D array of the size n_hidden*(features+1)
        w2 = np.random.uniform(-1.0, 1.0, size = self.n_output*(self.n_hidden + 1))
        w2 = w2.reshape(self.n_output, self.n_hidden + 1)
        return w1, w2

    def _sigmoid(self, z):
        # expit is equivalent of 1.0/(1.0 + np.exp(-z))
        return expit(z)

    def _sigmoid_gradient(self, z):
        #this is for the derivative of the sigmoid function
        sg =self._sigmoid(z)
        return sg*(1 - sg)

    def _add_bias_unit(self, X, how = 'column'):
        #The how parameter works in way we want our vector multiplication to be
        if how == 'column':
            #the datatype holding the bias unit is named X_new
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X

        elif how == 'row':
            X_new = np.ones((X.shape[0] + 1, X.shape[1]))
            X_new[:, 1:] = X

        else:
            raise AttributeError(' how must be column or row')
        return X_new

    
#  Get the understanding of the below three functions
    def _feedforward(self, X, w1, w2):
        a1 = self._add_bias_unit(X, how = 'column')
        z2 = w1.dot(a1.T)
        a2 = self._sigmoid(z2)
        a2 = self._add_bias_unit(a2, how = 'row')
        z3 = w2.dot(a2)
        a3 = self._sigmoid(z3)
        return a1, z2, a2, z3, a3

    def _L2_reg(self, lambda_, w1, w2):
        return (lambda_/2.0) * (np.abs(w1[:, 1:] ** 2)\ + np.sum(w2[:, 1:] ** 2))

    def _L1_reg(self, lambda_, w1, w2):
        return (lambda_/2.0) * (np.abs(w1[:, 1:]).sum()\ + np.abs(w2[:, 1:]).sum())

    def _get_cost(self, y_enc, output, w1, w2):
        term1 = -y_enc * (np.log(output))
        term2 = (1-y_enc) * (np.log(1-output))
        cost = np.sum(term1 - term2)
        L1_term = self._L1_reg(self.l1, w1, w2)
        L2_term = self._L2_reg(self.l2, w1, w2)
        cost = cost + L1_term + L2_term
        return cost
        
    def _get_gradient(self, a1, a2, a3, z2, y_enc, w1, w2):
        #backpropagation
        sigma3 = a3 - y_enc
        z2 = self._add_bias_units(z2, how = 'row')
        sigma2 = w2.T.dot(sigma3) * self._sigmoid_gradient(z2)
        sigma2 = sigma[1:, :]
        grad1 = sigma2.dot(a1)
        grad2 = sigma3.dot(a2.T)

        #regularize
        grad1[:, 1:] += (w1[:, 1:] * (self.l1 + self.l2))
        grad2[:, 1:] += (w2[:, 1:] * (self.l1 + self.l2))
        return grad1, grad2

    def predict(self, X):
        a1, z2, a2, z3, a3 = self._feedforward(X, self.w1, self.w2)
        y_pred = np.argmax(z3, axis = 0)
        return y_pred

    def fit(self, X, y, print_progress = False):
        self.cost = []
        X_data, y_data = X.copy(), y.copy()
        y_enc = self._encode_labels(y, self.n_output)

        delta_w1_prev = np.zeros(self.w1.shape)
        delta_w2_prev = np.zeros(self.w2.shape)

        for i in range(self.epochs) :

            #adaptive learning rate
            self.eta /= (1 + self.decrease_const*1)

            if print_progress:
                sys.stderr.write('\rEpoch: %d/%d' % (1+i, self.epochs))
                sys.stderr.flush()

            if self.shuffle:
                idx = np.random.permutation(y_data.shape[0])
                X_data, y_data = X_data[idx], y_data[idx]

            mini = np.array.split(range(y_data.shape[0]), self.minibatches)

            for idx in mini:

                a1, z2, a2, z3, a3 = self._feedforward(X[idx], self.w1, self.w2)
                cost = self._get_cost(y_enc = y_enc[:, idx] , output = a3, w1 =self.w1, w2 = self.w2)
                self.cost_.append(cost)

                grad1, grad2 = self._get_gradient(a1 = a1, a2=a2, a3=a3, z2=z2, y_enc =y_enc[:, idx],w1 = self.w1, w2 = self.w2)

                delta_w1, delta_w2 = self.eta*grad1, self.eta*grad2

                self.w1 -= (delta_w1 + (self.alpha * delta_w1_prev))
                self.w2 -= (delta_w2 + (self.alpha * delta_w2_prev))
                delta_w1_prev, delta_w2_prev = delta_w1, delta_w2

    return self
