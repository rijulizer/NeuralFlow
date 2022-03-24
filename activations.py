import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def tanh(x):
    return np.tanh(x)

# Derivatives
def sigmoid_prime(z):
    s = sigmoid(z)
    return s*(1-s)

def tanh_prime(x):
    return 1- np.tanh(x)**2