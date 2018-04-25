import _thread
import random
from math import ceil
from queue import Queue
from time import sleep

import numpy as np
import flappy
from bird import Bird


def create_first_population(population_length):
    # todo: bias are all zeros
    birds = []
    for _ in range(population_length):
        birds.append(Bird())

    return birds


def sort_birds_by_fitness(birds):
    return sorted(birds, key=lambda x: x.fitness, reverse=True)


def evolve_population(birds, best_birds=3):
    """

    :param birds: The birds
    :param best_birds: The number of birds to keep. Must be >= 2
    :return: array of birds
    """

    new_birds = []

    for i in range(len(birds)):

        # Keep the best birds
        if i < best_birds:
            new_birds.append(birds[i])

        # The rest of the room available is cut by 2
        # Crosses over with the best winners
        elif i < ceil((len(birds) - best_birds) / 2):
            parentA = birds[random.randint(0, best_birds) - 1]
            parentB = birds[random.randint(0, best_birds) - 1]
            new_birds.append(Bird.crossover(parentA, parentB))

            # Make a mutation
            new_birds[i].mutate()

        # Crosses over with a random
        else:
            parentA = birds[random.randint(0, len(birds)) - 1]
            parentB = birds[random.randint(0, len(birds)) - 1]
            new_birds.append(Bird.crossover(parentA, parentB))

            # Make a mutation
            new_birds[i].mutate()

        # Reset fitness
        new_birds[i].distance_traveled = 0
        new_birds[i].fitness = 0

    return new_birds


def start_flappy():
    flappy.main()


def reset_bird(bird):
    bird.fitness = 0
    bird.distance_traveled = 0
    bird.create_brain()


def main():
    # Starts game
    _thread.start_new_thread(start_flappy, ())
    generation = 1
    population = 8
    old_fitness_mean = 0
    new_fitness_mean = 0

    best_bird_ever = Bird()
    best_score_ever = 0

    birds = create_first_population(population)

    flappy.population = population

    while True:
        old_fitness_mean = new_fitness_mean
        for bird in birds:
            reset_bird(bird)
            # todo: tester de l'enlever
            #sleep(0.2)

        flappy.start()

        # todo: passer le tout en fonction
        # todo: passer le tout dans flappy.py et supprimer ici
        flappy.pooulation = population
        flappy.score = np.zeros(population)
        flappy.diff_x = np.zeros(population)
        flappy.diff_y = np.zeros(population)
        flappy.is_alive = np.full(population, True)

        # While at least one bird is alive
        while len(flappy.birds) > 0:
        #while np.all([flappy.is_alive, np.full(population, True)]):

            for i in range(len(birds)):

                if flappy.is_alive[i]:
                    bird = birds[i]

                    bird.increase_fitness(flappy.score[i])

                    # todo : tester avant de l'enlever
                    bird.should_flap(0, 0)

                    diff_x = flappy.diff_x[i]
                    diff_y = flappy.diff_y[i]
                    # todo : resoudre le problème des scores
                    prediction = bird.should_flap(diff_x, diff_y)
                    if prediction:
                        flappy.flap(i)

            sleep(0.01)

        # Updates array
        #birds[i] = bird
        #print("Generation {}: - Individual {}: - Fitness: {}".format(generation, i, bird.fitness))
        print("Score {}".format(flappy.score))
        birds = sort_birds_by_fitness(birds)

        # Code moche !!!!!!

        for bird in birds:
            new_fitness_mean += bird.fitness

        new_fitness_mean /= len(birds)

        print("Old: {} VS New {}".format(old_fitness_mean, new_fitness_mean))
        # todo: garder le meilleur oiseau de tout les temps
        # todo : fitness : score - A * diffX
        # todo: ne pas avoir le debug, spawn pipes direct

        if birds[0].fitness > best_score_ever:
            best_bird_ever = best_bird_ever
            best_score_ever = birds[0].fitness
            print("New best score with {} !".format(best_score_ever))

        if birds[0].fitness == 0:
            print("Starting new population. This one was too bad :(")
            #generation = 0

            best = birds[0]
            birds = create_first_population(population)
            birds[0] = best

        else:
            birds = evolve_population(birds)
            birds[population - 1] = best_bird_ever

        generation += 1


if __name__ == "__main__":
    # todo : set weights
    # print("bonjour")
    main()
