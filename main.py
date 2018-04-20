import _thread
import os
import random
from imp import reload
from multiprocessing import Process
from queue import Queue
from time import sleep, time
from typing import List

import pygame

import keras
from keras import Sequential
from keras.layers import Dense, np
from keras.optimizers import SGD
import flappy
from bird import Bird

import tensorflow as tf
from keras import backend


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

def sort_birds_by_fitness(birds):
    return sorted(birds, key=lambda x: x.fitness, reverse=True)

def start_flappy():
    flappy.main()



def observe_bird(bird: Bird, queue: Queue):

    while flappy.is_alive:
        bird.increase_fiteness()
        prediction = bird.should_flap(flappy.diff_x, flappy.diff_y)
        print(prediction)
        if prediction:
            flappy.flap()
        sleep(0.05)

    print(bird.fitness)
    queue.put(bird)


def main():
    _thread.start_new_thread(start_flappy, ())
    birds = create_first_population()
    for i in range(3):

        session = tf.Session()
        backend.set_session(session)
        backend.get_session().run(tf.global_variables_initializer())
        sleep(1)

        bird = birds[i]
        bird.create_brain()

        queue = Queue()

        # Tensorflow error without sleep
        sleep(1)

        flappy.start()

        # Does nothing on the bird beahivour but Keras crashes if prediction is not used in the main thread at least once
        bird.should_flap(0, 0)

        _thread.start_new_thread(observe_bird, (bird, queue))
        bird = queue.get()
        birds[i] = bird

        # Updates bird
        #print(bird.fitness)
        sleep(1)

    sortedBirds = sort_birds_by_fitness(birds)
    for bird in sortedBirds:
        print(bird.fitness)

if __name__ == "__main__":
    main()
    #_thread.start_new_thread(start_flappy, ())
