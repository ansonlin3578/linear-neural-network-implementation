from ctypes import create_unicode_buffer
from turtle import forward
import numpy as np
import pdb
class full_connected(object):
    def __init__(self, input_size , output_size , momentum) -> None:    #fc1=(784,256) , fc2=(256,128) , fc3=(128,10)
        self.weights = np.random.randn(input_size , output_size) / input_size   #normalization
        self.bias = np.zeros(output_size)
        self.momentum = momentum
        self.adj_weights = np.zeros_like(self.weights)
        self.adj_bias = np.zeros_like(self.bias)
    
    def forward(self, input_data):  
        self.last_data = input_data
        result = np.dot(input_data , self.weights) + self.bias
        return result   #result_size = (32,256) -> (32,128) -> (32,10)

    def backward(self, loss_in , lr):
        batch_size = loss_in.shape[0]   
        grad_weights = np.dot(self.last_data.T , loss_in) / batch_size    #normalization
        grad_bias = np.sum(loss_in , axis=0) / batch_size
        grad = np.dot(loss_in , self.weights.T)

        self.adj_weights = self.adj_weights * self.momentum - lr * grad_weights
        self.adj_bias = self.adj_bias * self.momentum - lr * grad_bias
        self.weights += self.adj_weights
        self.bias += self.adj_bias
        return grad    #size : (32,128) -> (32,256) -> (32,784) 

class flatten(object):
    def forward(self, input_data):
        self.batch_size , self.input_channel , self.x , self.y = input_data.shape
        return input_data.reshape(self.batch_size , self.input_channel * self.x * self.y)

class relu(object):
    def forward(self, input_data):  
        self.last_val = input_data
        result = input_data.copy()
        result[result < 0] = 0
        return result
    def backward(self, residual):
        current_grad = residual.copy()
        current_grad[self.last_val < 0] = 0
        return current_grad

class sigmoid(object):
    def forward(self, input_data):
        self.last_val = input_data
        result = input_data.copy()
        result = (1.0 / (1.0 + np.exp(np.negative(result))))
        return result

    def backward(self, residual):
        prev_grad = residual.copy()
        current_grad = prev_grad * (1.0 / (1.0 + np.exp(np.negative(self.last_val)))) * (1 - (1.0 / (1.0 + np.exp(np.negative(self.last_val)))))
        return current_grad

class tanh(object):
    def forward(self, input_data):
        self.last_val = input_data
        result = input_data.copy()
        result = np.tanh(result)
        return result
        
    def backward(self, residual):
        prev_grad = residual.copy()
        current_grad = prev_grad * (1 - np.tanh(self.last_val)**2)
        return current_grad

class softmax(object):
    def forward(self, input_data):  #input_data.shape : (32, 10)
        self.result = np.zeros((input_data.shape))
        data_batch = input_data.shape[0]
        for i in range(data_batch):
            self.expo = np.exp(input_data[i])
            temp = self.expo / np.sum(self.expo)
            self.result[i] = temp
        return self.result     

    def backward(self, label_input):
        return self.result - label_input    #size (32,10)

class cross_entropy(object):
    def forward(self, input_data , input_label):
        accuracy = 0
        data_batch = input_data.shape[0]
        for i in range(data_batch):
            if (np.argmax(input_data[i]) == np.argmax(input_label[i])):
                accuracy += 1
        loss_temp = -np.multiply(input_label , np.log(input_data))
        loss_result = np.sum(loss_temp)

        return accuracy , loss_result , np.argmax(input_label , axis = 1)
