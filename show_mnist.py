from matplotlib import pyplot as plt
import numpy as np
from hw1 import Get_mnist_data
import gzip

#fashion mnist dataset path      
url_train_image = 'Fashion_MNIST_data/train-images-idx3-ubyte.gz'
url_train_labels = 'Fashion_MNIST_data/train-labels-idx1-ubyte.gz'
url_test_image = 'Fashion_MNIST_data/t10k-images-idx3-ubyte.gz'
url_test_labels = 'Fashion_MNIST_data/t10k-labels-idx1-ubyte.gz'

#use gzip open .gz to get ubyte
train_image_ubyte = gzip.open(url_train_image,'r')
test_image_ubyte = gzip.open(url_test_image,'r')
train_label_ubyte = gzip.open(url_train_labels,'r')
test_label_ubyte = gzip.open(url_test_labels,'r')
obj = Get_mnist_data()
dataset = {
    'training':{
        "data" : obj.load_data('train')
        ,"label" : obj.load_data('train_label')
                }
    ,"testing":{
        "data" : obj.load_data('test')
        ,"label" : obj.load_data('test_label')
                }
            }

first_image = dataset["training"]["data"][0]
first_image = np.array(first_image, dtype='float')
pixels = first_image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()