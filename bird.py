import random

from keras import Sequential
from keras.layers import Dense, np
from keras.optimizers import SGD


class Bird:

    def __init__(self):
        self.model = None
        self.fitness = 0
        self.index = 0

    def create_brain(self):
        """
        Creates the bird's neural network
        inputs is an array
        """

        self.model = Sequential()  # Model Creation

        # Create a hidden layer with 2 input layers and 16 neurons and relu activation
        self.model.add(Dense(units=16, input_dim=2, activation="relu"))

        # Create the output layer
        self.model.add(Dense(units=1, activation='sigmoid'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss="mse", optimizer=sgd, metrics=["accuracy"])

    def crossover(cls, bird1, bird2):
        """
        Static method
        Exchanges the hidden layer input weights
        Note: Sequencial.getweights() gives [layer 1[weight input 1, weight input 2], layer2[weight input 1]
        """

        # Gets the neural network
        model1: Sequential = bird1.model
        model2: Sequential = bird2.model

        # Get all the weights of the first layer
        weights1 = model1.get_weights()[0]
        weights2 = model2.get_weights()[0]

        weightnew1 = weights1
        weightnew2 = weights2

        # Crosses over the first input weight with the second input weights
        weightnew1[0] = weights2[0]
        weightnew2[0] = weights1[0]

        return np.array([weightnew1, weightnew2])

    def mutate(cls, Bird, mutation_probability=0.2, mutation_strength=0.5):

        model: Sequential = Bird.model

        weights = model.get_weights()

        for layer in range(len(weights)):
            inputs = weights[layer]

            for weight in range(len(inputs)):

                if random.random() <= mutation_probability:
                    mutation = random.uniform(- mutation_strength, + mutation_strength)
                    weights[layer][weight] += mutation

        return weights

    def should_flap(self, diff_x, diff_y):
        # Converts the array in 2D: Keras needs a 2 dimensional array
        # Dim 1: The items array. The length is always 1 in this case
        # Dim 2: The input length.
        inputs = np.atleast_2d([diff_x, diff_y])

        prediction = self.model.predict(inputs)
        return prediction > 0.5

    def increase_fiteness(self):
        self.fitness += 1

    crossover = classmethod(crossover)
    mutate = classmethod(mutate)
