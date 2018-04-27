import random
from neuralnetwork import NeuralNetwork


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
            self.model = NeuralNetwork()
            self.model.create_brain()

    @staticmethod
    def crossover(birdA, birdB):

        # todo a optimiser
        biasesA = birdA.model.hidden_layer[1]
        biasesB = birdB.model.hidden_layer[1]

        cut = random.randint(0, len(biasesA))

        for i in range(cut):
            oldBiasA = biasesA[i]
            biasesA[i] = biasesB[i]
            biasesB[i] = oldBiasA

        birdA.model.hidden_layer[1] = biasesA
        birdB.model.hidden_layer[1] = biasesB

        # Gets all the weights of the first layer
        birdA.model.hidden_layer[1] = biasesA
        birdB.model.hidden_layer[1] = biasesB

        if random.random() < 0.5:
            return birdA

        else:
            return birdB

    # noinspection PyShadowingBuiltins
    def mutation(self, mutation_probability=0.2):

        # Mutate the hidden and output layer
        for i in range(NeuralNetwork.hidden_neurons):

            # Mutate the hidden layer first input
            self.model.hidden_layer[0][0][i] = Bird.mutate(self.model.hidden_layer[0][0][i], mutation_probability)

            # Mutate the hidden layer second input
            self.model.hidden_layer[0][1][i] = Bird.mutate(self.model.hidden_layer[0][1][i], mutation_probability)

            # Mutate the hidden layer biases
            self.model.hidden_layer[1][i] = Bird.mutate(self.model.hidden_layer[1][i], mutation_probability)

            # Mutate the output layer input
            self.model.output_layer[0][0] = Bird.mutate(self.model.output_layer[0][0], mutation_probability)

        # Mutate the output layer bias
        self.model.output_layer[1] = Bird.mutate(self.model.output_layer[1], mutation_probability)

    @staticmethod
    def mutate(weight, mutation_probability):

        if random.random() < mutation_probability:
            mutationIntensity = 1 + ((random.random() - 0.5) * 3 + (random.random() - 0.5))
            weight *= mutationIntensity

        return weight

    def should_flap(self, diff_x, diff_y):
        # Converts the array in 2D: Keras needs a 2 dimensional array
        # Dim 1: The items array. The length is always 1 in this case
        # Dim 2: The input length.
        prediction = self.model.predict(diff_x, diff_y)
        return prediction > 0.5

    def increase_fitness(self, score, diffx):

        self.distance_traveled += 1
        self.fitness = score - diffx / 1000
        self.score = score
        # self.fitness = 5 * self.distance_traveled - diff_x + 1000 * score
        # self.fitness -= diff_x / 20
        # self.fitness += 10 * 1 / (diff_y * diff_y * 5 + 1) # + self.distance_traveled
        # self.fitness += 5 - (1 / 5000) * pow(diff_y, 2)
