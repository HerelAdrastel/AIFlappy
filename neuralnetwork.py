import math

import numpy as np

class NeuralNetwork:
    
    hidden_neurons = 16

    def __init__(self):

        # Shape of hidden layer:
        # [[weights1, weights2], bias]
        self.hidden_layer = []

        # Shape of output layer
        self.output_layer = []

    @staticmethod
    def relu(nb):
        return nb * (nb > 0)

    @staticmethod
    def sigmoid(nb):
        return 1 / (1 + np.exp(-nb))

    def create_brain(self):
        # Create weights for hidden layer
        w1 = np.random.uniform(-1, 1, NeuralNetwork.hidden_neurons)
        w2 = np.random.uniform(-1, 1, NeuralNetwork.hidden_neurons)
        biases = np.random.uniform(-1, 1, NeuralNetwork.hidden_neurons)

        self.hidden_layer = np.array([[w1, w2], biases])

        # Creates weights for output layer
        w = np.random.uniform(-1, 1, NeuralNetwork.hidden_neurons)
        bias = np.random.uniform(-1, 1)
        self.output_layer = np.array([[w], bias])

    def predict(self, diffX, diffY):
        prediction = diffX * self.hidden_layer[0][0] + diffY * self.hidden_layer[0][1] + self.hidden_layer[1]
        prediction = NeuralNetwork.relu(prediction)

        prediction = np.sum(prediction * self.output_layer[0][0]) + self.output_layer[1]
        prediction = NeuralNetwork.sigmoid(prediction)

        return prediction