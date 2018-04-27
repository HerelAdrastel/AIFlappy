import math

import numpy as np

class NeuralNetwork:

    def __init__(self):

        self.inputLayer = []
        self.hiddenLayer = []
        self.outputLayer = []

        self.hidden_neurons = 16

    @staticmethod
    def sigmoid(nb):
        return 1 / (1 + np.exp(-nb))

    def create_brain(self):
        # Create weights for hidden layer
        w1 = np.random.uniform(-1, 1, self.hidden_neurons)
        w2 = np.random.uniform(-1, 1, self.hidden_neurons)
        biases = np.random.uniform(-1, 1, self.hidden_neurons)

        self.hiddenLayer = np.array([[w1, w2], biases])

        # Creates weights for output layer
        w = np.random.uniform(-1, 1, self.hidden_neurons)
        bias = np.random.uniform(-1, 1)
        self.outputLayer = np.array([[w], bias])

    def predict(self, diffX, diffY):
        prediction = diffX * self.hiddenLayer[0][0] + diffY * self.hiddenLayer[0][1] + self.hiddenLayer[1]
        prediction = NeuralNetwork.sigmoid(prediction)

        prediction = np.sum(prediction * self.outputLayer[0][0]) + self.outputLayer[1]
        prediction = NeuralNetwork.sigmoid(prediction)

        return prediction