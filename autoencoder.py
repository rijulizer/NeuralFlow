import numpy as np
from network import Network
from layers import Dense, ActivationLayer
from activations import Activations #tanh,sigmoid,tanh_prime,sigmoid_prime
from loss import mse, mse_prime, binary_cross_entropy, bce_prime


# training data
x_train = np.array(
    [[0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,1,0],
    [0,0,0,0,0,1,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,1,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,1,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0]]
)
# reshape the input 
x_train = np.reshape(x_train,(8,8,1))
print(x_train.shape) #(8,8,1)
y_train = x_train


# define the auto encoder network
net = Network()
net.add(Dense(8,3,'FC_1'))
net.add(Dense(3,8,'FC_2'))
# add sigmoid adctivation in the last layer
net.add(ActivationLayer('sigmoid', 'activ_1'))
# use MSE as the loss
net.use_loss(mse,mse_prime)

# Train
net.fit(x_train, y_train, epochs=10000, learning_rate=0.1)

# predict
print('testing with training data')
out = net.predict(x_train)
print(out)