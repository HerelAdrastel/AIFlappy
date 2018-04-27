import _thread
import random
import sys
from os import system
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


def evolve_population(birds, best_birds=4):
    """

    :param birds: The birds
    :param best_birds: The number of birds to keep. Must be >= 2
    :return: array of birds
    """

    new_birds = []

    if len(birds) != 10:
        raise Exception("Bird's length must be equal to 10")

    # Keep the N best_birds (4 by default)
    for i in range(best_birds):
        new_birds.append(birds[i])

    # Make a crossover of the 2 best winners
    new_birds.append(Bird.crossover(birds[0], birds[1]))

    # Makes 3 crossovers of random best_birds (4 by default)
    for i in range(3):
        birdA = birds[random.randint(0, best_birds - 1)]
        birdB = birds[random.randint(0, best_birds - 1)]

        new_birds.append(Bird.crossover(birdA, birdB))

    # Copies one bird among the bests
    for i in range(2):
        new_birds.append(birds[random.randint(0, best_birds - 1)])

    # Resets bird and mutate
    new_birds[9].mutation()

    for bird in new_birds:
        bird.fitness = 0
        bird.score = 0
        bird.distance_traveled = 0

    return new_birds


def start_flappy():
    flappy.main()


def arg():
    if len(sys.argv) > 1:
        print(sys.argv[1])
        return float(sys.argv[1])

    return 1


def reset_bird(bird):
    bird.fitness = 0
    bird.distance_traveled = 0
    bird.create_brain()


def save(array, name):
    system("touch {}.txt".format(name))
    with open("{}.txt".format(name), 'w') as file:
        for item in array:
            file.write("{}\n".format(item))


def main():
    # Starts game
    _thread.start_new_thread(start_flappy, ())

    generation = 1
    population = 10

    best_bird_ever = Bird()
    best_score_ever = 0

    birds = create_first_population(population)

    flappy.population = population

    fitnesses = np.array([])


    while True:

        for bird in birds:
            reset_bird(bird)

        # todo: passer le tout en fonction
        # todo: passer le tout dans flappy.py et supprimer ici
        flappy.population = population
        flappy.score = np.zeros(population)
        flappy.diff_x = np.zeros(population)
        flappy.diff_y = np.zeros(population)
        flappy.is_alive = np.full(population, True)
        flappy.birds = np.arange(population)

        sleep(1)
        flappy.start()
        # While at least one bird is alive
        while len(flappy.birds) > 0:

            for i in range(len(birds)):

                if flappy.is_alive[i]:
                    bird = birds[i]

                    # todo : tester avant de l'enlever
                    bird.should_flap(0, 0)

                    diff_x = flappy.diff_x[i]
                    diff_y = flappy.diff_y[i]
                    bird.increase_fitness(flappy.score[i], diff_x)
                    # todo : resoudre le problème des scores
                    prediction = bird.should_flap(diff_x, diff_y)
                    if prediction:
                        flappy.flap(i)

            sleep(0.01)

        birds = sort_birds_by_fitness(birds)

        fitnesses = np.append(fitnesses, np.max([bird.fitness for bird in birds]))

        print("Generation {} - Best {} - Median {}".format(generation, birds[0].fitness,
                                                           np.median([bird.fitness for bird in birds])))


        if birds[0].fitness > best_score_ever:
            best_bird_ever = best_bird_ever
            best_score_ever = birds[0].fitness
            print("New best score with {} !".format(best_score_ever))

        if birds[0].score <= 0:
            print("Starting new population. This one was too bad :(\n")
            # generation = 0
            sleep(1)
            best = birds[0]
            birds = create_first_population(population)
            birds[0] = best



        else:
            print("Evolving population...\n")
            birds = evolve_population(birds)
            birds[population - 1] = best_bird_ever

        save(fitnesses, "save")
        generation += 1


if __name__ == "__main__":
    # todo : set weights
    # print("bonjour")
    main()
