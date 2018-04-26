import random
from time import sleep

import numpy as np
import matplotlib.pyplot as plt


class Graph:

    def __init__(self):
        self.x = 0
        self.y = 0
        self.i = 0
        self.xscale = 10
        self.yscale = 10

        fig = plt.figure(1)
        self.ax = fig.add_subplot(111)
        self.ax.set_xlim(0, self.xscale)
        self.ax.set_ylim(0, 10)
        self.line, = self.ax.plot(self.x, self.y, 'ko-')
        plt.pause(0.001)

    def update(self, y):
        self.i += 1

        if self.i > self.xscale -2:
            self.xscale += 10
            self.ax.set_xlim(0, self.xscale)

        if np.max(y) > self.yscale - 2:
            self.yscale += 10
            self.ax.set_ylim(0, self.yscale)

        self.x = np.arange(len(y))
        self.y = y

        self.line.set_data(self.x, self.y)


if __name__ == '__main__':

    gra = Graph()
    a = np.array([])

    a = np.append(a, random.uniform(0, 20))
    gra.update(a)
