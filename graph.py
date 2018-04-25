import matplotlib.pyplot as plt
        import numpy as np

class Graph:

    def __init__(self):


        plt.ion()
        fig, ax = plt.subplots()
        x, y = [], []
        sc = ax.scatter(x, y)
        plt.xlim(0, 10)
        plt.ylim(0, 10)

        plt.draw()
        scale = 10