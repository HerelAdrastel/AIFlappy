import numpy as np


class Connection:
    def __init__(self, connectedNeuron):
        self.connectedNeuron = connectedNeuron
        self.weight = np.random.normal()
        self.dWeight = 0.0


class Neuron:

    def __init__(self, layer):
        self.dentrites = []
        self.output = 0.0
        if layer is None:
            pass
        else:
            for neuron in layer:
                connection = Connection(neuron)
                self.dentrites.append(connection)