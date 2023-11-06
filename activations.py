import numpy as np

class Activations:
    def __init__(self):
        self.activ_functions = {
            'sigmoid': self.sigmoid,
            'relu': self.relu,
            'tanh': self.tanh,
            'softmax': self.softmax,
        }
        self.activ_functions_prime = {
            'sigmoid': self.sigmoid_prime,
            'relu': self.relu_prime,
            'tanh': self.tanh_prime,
            'softmax': self.softmax_prime,
        }
    def get_activation_function(self, name: str):
        return self.activ_functions[name]

    def get_activation_prime(self, name: str):
        return self.activ_functions_prime[name]
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def relu(self, z):
        return np.maximum(0, z)

    def tanh(self, x):
        return np.tanh(x)
    
    def softmax(self, z):
        x = np.exp(z)-max(z)
        return x / x.sum(axis=0)


    # Derivatives
    def sigmoid_prime(self, z):
        s = self.sigmoid(z)
        return s*(1-s)

    def tanh_prime(self, z):
        return 1- np.tanh(z)**2

    def relu_prime(self, z):
        return NotImplementedError
    
    def softmax_prime(self, z):
        """
        referred: https://stats.stackexchange.com/questions/215521/how-to-find-derivative-of-softmax-function-for-the-purpose-of-gradient-descent
        """
        return NotImplementedError


if __name__=="__main__":
    actvations = Activations()
    print(actvations.get_activation_function('sigmoid')(5.0))
    print(actvations.get_activation_prime('sigmoid')(5.0))
    print(actvations.get_activation_function('softmax')([5.0,6.0,13.5]))
    # print(actvations.get_activation_prime('s')(5.0))
    