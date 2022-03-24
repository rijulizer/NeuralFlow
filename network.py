class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self,layer):
        self.layers.append(layer)
    
    def use_loss(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime
    
    def predict(self, input_data):
        
        result = []

        # iterate through data points
        for i in range(len(input_data)):
            output = input_data[i]
            # iterate through the layers
            for layer in self.layers:
                output = layer.forward(output)
            # append result of final layer output
            result.append(output)
        return result
    
    def fit(self, x_train, y_train, epochs, learning_rate):

        for i in range(epochs):
            # print(f'epoch - {i} ')
            forward_err = 0
            for j in range(len(x_train)):
                # forward propagation
                output = x_train[j]
                # print(f'input is -> {output}')
                for layer in self.layers:
                    output = layer.forward(output)
                
                forward_err += self.loss(y_train[j], output)

                # backward propagation
                backward_error = self.loss_prime(y_train[j], output)
                # print(f'last layer error{backward_error}')
                for layer in reversed(self.layers):
                    backward_error = layer.backward_propagation(backward_error, learning_rate)
                    # print(f'at layer {layer}, backward error -> {backward_error}')
            if i % 10 ==0:

                forward_err /= len(x_train)
                print(f'epoch {i+1}/{epochs}  error ={forward_err}')