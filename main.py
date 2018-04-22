import _thread
from queue import Queue
from time import sleep

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


def evolve_population(population):

    # If all birds coudn't read halt distance of the first barrier, kill them all
    if population[0].fitness < 0:
        return create_first_population(len(population))

    new_weights = []

    new_weights.append(population[0].model.get_weights())
    new_weights.append(population[1].model.get_weights())
    new_weights.append(Bird.crossover(population[0], population[1]))
    new_weights.append(Bird.crossover(population[1], population[0]))

    for i in range(4, len(population)):
        new_weights.append(population[i].model.get_weights())

    for i in range(len(population)):

        # Update weights
        population[i].model.set_weights(new_weights[i])
        population[i].mutate()
        population[i].fitness = 0


    del population[-1]
    population.append(Bird())

    return population


def start_flappy():
    flappy.main()



def main():
    _thread.start_new_thread(start_flappy, ())
    birds = create_first_population(2)

    generation = 1

    while True:
        for i in range(len(birds)):
            bird = birds[i]
            bird.create_brain()

            queue = Queue()

            # Tensorflow error without sleep
            sleep(0.2)

            flappy.start()
            # Does nothing on the bird beahivour but Keras crashes if prediction is not used in the main thread at least once
            bird.should_flap(0, 0)

            flappy.is_alive = True
            flappy.score = 0
            flappy.diff_x = 0
            flappy.diff_y = 0

            while flappy.is_alive:
                diff_x = flappy.diff_x
                diff_y = flappy.diff_y
                bird.increase_fiteness(diff_x)
                prediction = bird.should_flap(diff_x, diff_y)
                if prediction:
                    flappy.flap()
                sleep(0.05)


            # Updates array
            birds[i] = bird
            print("Generation {}: - Individual {}: - Fitness: {}".format(generation, i, bird.fitness))
            input()

        birds = sort_birds_by_fitness(birds)
        evolve_population(birds)

        generation += 1


if __name__ == "__main__":
    # todo : set weights
    # print("bonjour")
    main()
