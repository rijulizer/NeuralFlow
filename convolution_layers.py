import numpy as np
from scipy import signal
from layers import Layer

class Conv2d(Layer):
    def __init__(self, input_shape, kernel_size, kernels, name=None):
        input_depth, input_height, input_width = input_shape
        #number of kernels
        self.kernels = kernels
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (kernels, input_height- kernel_size +1, input_width - kernel_size + 1)
        self.kernels_shape = (kernels, input_depth, kernel_size, kernel_size)
        self.name = name
        
        # Initiate kernels and biases
        self.kernel = np.random.rand(*self.kernels_shape) - 0.5 # (4,3,2,2)
        self.bias = np.random.rand(*self.output_shape) - 0.5

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.bias) 
        for i in range(self.kernels):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernel[i,j],'valid')
            
        return self.output 

    def backward_propagation(self, output_error, learning_rate):
        # output_error is the gradient of error wrt. the output of this model
        self.output_error = output_error
        # Initiate gradients, gradients must have same shape as kernels
        self.kernel_gradients = np.zeros(self.kernels_shape)
        self.input_gradients = np.zeros(self.input_shape)
        for i in range(self.kernels):
            for j in range(self.input_depth):
                self.kernel_gradients[i] = signal.correlate2d(self.input[j], self.output_error[i],'valid')
                self.input_gradients[j] += signal.convolve2d(self.output_error[i], self.kernel[i,j],'full')

        self.kernel -= learning_rate * self.kernel_gradients
        self.bias -= learning_rate * self.output_error        
        return self.input_gradients

class Reshape(Layer):
    def __init__(self, input_shape, output_shape, name=None):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.name = name

    def forward(self, input):
        return np.reshape(input, self.output_shape)
    
    def backward_propagation(self, output_error, learning_rate):
        return np.reshape(output_error, self.input_shape)
