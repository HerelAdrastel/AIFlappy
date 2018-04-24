import random

from keras import Sequential
from keras.layers import Dense, np
from keras.optimizers import SGD


class Bird:

    def __init__(self):
        self.model = None
        self.fitness = 0
        self.index = 0
        self.distance_traveled = 0
        self.score = 0

    def create_brain(self):
        """
        Creates the bird's neural network
        inputs is an array
        """
        if self.model is None:
            self.model = Sequential()  # Model Creation

            # Create a hidden layer with 2 input layers and 16 neurons and relu activation
            self.model.add(Dense(units=16, input_dim=2, activation="relu"))

            # Create the output layer
            self.model.add(Dense(units=1, activation='sigmoid'))

            sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            self.model.compile(loss="mse", optimizer=sgd, metrics=["accuracy"])
            self.model._make_predict_function()

            # By default, all biases are set to 0
            # Set all biases to a random number
            for layer in self.model.layers:
                weights = layer.get_weights()
                new_biases = np.random.uniform(size=len(weights[1]))

                weights[1] = new_biases
                layer.set_weights(weights)

    def crossover(cls, birdA, birdB):
        """
        Static method
        Exchanges the hidden layer input weights
        The Keras model organisation
        Layers:
            Weights
                Input 1
                Input 2
            Biases
        OU
        Weights
            Layers

        Examples:
            To get biases of the first layer : model.layers[0].get_weights()[1]
            To get the weights of the first input of the first layer : model.layer[0].get_weights()[0][0]
        """

        # Gets all the weights of the first layer
        weightsA = birdA.model.layers[0].get_weights()
        weightsB = birdB.model.layers[0].get_weights()

        # Get all the biases of the first layer
        biasesA = weightsA[1]
        biasesB = weightsB[1]

        # Determines where we must make the separation of the genome
        cut = random.randint(0, len(biasesA))

        for i in range(cut):
            oldBiasA = biasesA[i]
            biasesA[i] = biasesB[i]
            biasesB[i] = oldBiasA

        # Updates the weights
        weightsA[1] = biasesA
        weightsB[1] = biasesB

        birdA.model.set_weights(weightsA)
        birdB.model.set_weights(weightsB)

        # todo: optimisation possible
        if random.random() < 0.5:
            return birdA

        else:
            return birdB

    def mutate(self, mutation_probability=0.2, mutation_strength=0.3):

        for layer in self.model.layers:
            infos = layer.get_weights()

            weights = infos[0]
            biases = infos[1]

            for inputs in range(len(weights)):
                for input in range(len(weights[inputs])):
                    new_weight = Bird.mutate_weight(input, mutation_probability, mutation_strength)
                    layer.get_weights()[0][inputs][input] = new_weight

            for bias in range(len(biases)):
                new_bias = Bird.mutate_weight(bias, mutation_probability, mutation_strength)
                layer.get_weights()[1][bias] = new_bias


    def mutate_weight(cls, weight, mutation_probability, mutation_strength):

        if random.random() < mutation_probability:
            mutationIntensity = 1 + random.uniform(- mutation_strength, mutation_strength)
            weight *= mutationIntensity

        return weight

    def should_flap(self, diff_x, diff_y):
        # Converts the array in 2D: Keras needs a 2 dimensional array
        # Dim 1: The items array. The length is always 1 in this case
        # Dim 2: The input length.
        inputs = np.atleast_2d([diff_x, diff_y])

        prediction = self.model.predict(inputs)
        return prediction > 0.5

    def increase_fitness(self, score):

        self.distance_traveled += 1
        self.fitness = score
        #self.fitness = 5 * self.distance_traveled - diff_x + 1000 * score
        # self.fitness -= diff_x / 20
        #self.fitness += 10 * 1 / (diff_y * diff_y * 5 + 1) # + self.distance_traveled
        #self.fitness += 5 - (1 / 5000) * pow(diff_y, 2)


    crossover = classmethod(crossover)
    mutate_weight = classmethod(mutate_weight)
