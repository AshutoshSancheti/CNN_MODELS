#To load data, run the following in the shell
# X_train, y_train = load_mnist('C:\Python27\_Machine_Learning\MNIST_dataset', kind='train')
# X_test, y_test = load_mnist('C:\Python27\_Machine_Learning\MNIST_dataset', kind='t10k')
# Its necessary to put your full path

import os
import struct
import numpy as np
def load_mnist(path, kind='train'):
    
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,'%s-labels.idx1-ubyte'% kind)
    images_path = os.path.join(path,'%s-images.idx3-ubyte'% kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
        
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
    return images, labels
