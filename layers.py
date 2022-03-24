import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self,input):
        raise NotImplementedError
    
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError

class Dense(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size, name=None):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
        self.name = name
        # print(f'layer ->{name} initiated Weights - >{self.weights}')

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        # Update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error

class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime, name=None):
        self.activation = activation
        self.activation_prime = activation_prime
        self.name = name

    def forward(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        input_error = output_error * self.activation_prime(self.input)
        return input_error

        