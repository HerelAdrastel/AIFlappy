import _thread
import os
import random
from time import sleep, time
from typing import List

from keras import Sequential
from keras.layers import Dense, np
from keras.optimizers import SGD
import flappy
from bird import Bird


def activate_brain():
    """ Creates the bird's neural network """

    model = Sequential()  # Model Creation

    # Create a hidden layer with 2 input layers and 16 neurons and relu activation
    model.add(Dense(units=16, input_dim=2, activation="relu"))

    # Create the output layer
    model.add(Dense(units=1, activation='sigmoid'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="mse", optimizer=sgd, metrics=["accuracy"])

    # Example of inputs
    inputs = np.array([2, 1])

    # Converts the array in 2D: Keras needs a 2 dimensional array
    # Dim 1: The items array. The length is always 1 in this case
    # Dim 2: The input length.
    inputs = np.atleast_2d(inputs)

    prediction = model.predict(inputs)
    print(prediction)


def create_first_population(population_length=8):
    birds = []
    for _ in range(population_length):
        birds.append(Bird())

    return birds


i = 0
birds = []


def start_flappy():
    global i, birds
    for _ in birds:
        flappy.main()

        i += 1

    birds.sort(key=lambda x: x.fitness, reverse=True)
    for bird in birds:
        print(bird.index)
    # os.system("python flappy.py")


def count_flappy():
    global birds
    while True:
        # X nearest pipe: pipemidpos
        # print("{}".format(flappy.diff_x))

        bird = birds[i]

        # CODE DEGEULASSE A VIRER
        bird.index = i

        if bird.should_flap(flappy.diff_x, flappy.diff_y):
            flappy.flap()

        sleep(0.05)


def main():
    global birds
    birds = create_first_population()

    #_thread.start_new_thread(start_flappy, ())
    #_thread.start_new_thread(count_flappy, ())

    s = birds[i].should_flap(50, 50)
    print(s)


if __name__ == "__main__":
    main()
