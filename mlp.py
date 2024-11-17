import random

class MLP:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
    
    def initialize(self):
        layers = []
        all_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        
        for i in range(len(all_sizes) - 1):
            fan_in = all_sizes[i]
            fan_out = all_sizes[i+1]
            
            limit = (2 / (fan_in))**0.5
            W = [[random.randint(-limit, limit) for _ in range(fan_out)] for _ in range(fan_in)]
            B = [0] * fan_out
            layers.append(W, B)

        self.layers = layers
        self.num_layers = len(layers)

    def ReLU(x):
        return max(0, x)

    def derivitive_ReLU(self, x):
        if x > 0:
            return 1
        return 0
    
    def dot_product(W, act):
        total = 0
        for j in range(len(W)):
                total += (W[j] * act[j])
        return total

    def forward_propogation(self, X):
        activations = [X]
        for i in range(1, len(self.layers)):
            W, B = self.layers[i-1]
            Z = self.dot_product(W, activations[i])
            for i in range(len(B)):
                Z[i] += B[i]
            A = self.ReLu(Z)
            X.append(A)
        
        return activations

    def loss(Y_True, Y_Pred):
        MSE = 0
        for i in range(len(Y_True)):
            MSE += (Y_True[i] - Y_Pred)**2

        return (MSE / len(Y_True))

    def back_propagation(self, X, Y, activations):
        dz = []
        for i in range(len(activations[-1])):
            dz.append(activations[i] - Y[i])
        
        dw = (self.dot_product(activations[-2], dz))/len(Y)
        db = sum(dz)/len(Y)

        for i in range(self.num_layers - 2):
            W_next = self.layers[i+1][0]
            dA = self.dot_product(dz[i], W_next)
            dZ = dA * self.derivitive_ReLU(activations[i+1])
            dW = (self.dot_product(activations[i], dZ))/len(Y)
            dB = sum(dZ)/len(Y)

        

        pass


