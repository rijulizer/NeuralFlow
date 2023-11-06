import numpy as np
from activations import Activations
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self,input):
        raise NotImplementedError
    
    def backward_propagation(self, output_gradient, learning_rate):
        raise NotImplementedError

class Dense(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size, name=None):
        self.weights = np.random.rand(output_size, input_size) - 0.5
        self.bias = np.random.rand(output_size, 1) - 0.5
        self.name = name
        print(f'layer ->{name} initiated Weights shape- >{np.shape(self.weights)}')

    def forward(self, input):
        """
        input = X matrix
        """
        self.input = input
        # print(f'layer ->{self.name} input shape- >{np.shape(self.input)}')
        self.output = np.dot(self.weights, self.input) + self.bias
        # print(f'layer ->{self.name} output shape- >{np.shape(self.output)}')
        
        return self.output
    
    def backward_propagation(self, output_gradient, learning_rate):
        """
        Given dE/dY derivative of error wrt. the output calculate - 
        dE/dW: dE/dY * X^t
        dE/dB: dE/dY 
        dE/dW: W^t * dE/dY 
        return the error wrt. to the input which can be back propagated
        """
        # calculate gradients
        weights_grad = np.dot(output_gradient, self.input.T)
        bias_grad = output_gradient
        input_grad = np.dot(self.weights.T, output_gradient)
        
        # Update parameters
        self.weights -= learning_rate * weights_grad
        self.bias -= learning_rate * bias_grad
        return input_grad

class ActivationLayer(Layer):
    def __init__(self, activation_fn_name, name=None):
        actvations = Activations()
        
        # self.activation = activation
        # self.activation_prime = activation_prime
        
        self.activation = actvations.get_activation_function(activation_fn_name)
        self.activation_prime = actvations.get_activation_prime(activation_fn_name)
        
        self.name = name

    def forward(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output
    
    def backward_propagation(self, output_gradient, learning_rate):
        # element wise multiplications
        input_grad = np.multiply(output_gradient, self.activation_prime(self.input))
        return input_grad

        