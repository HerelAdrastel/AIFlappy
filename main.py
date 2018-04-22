import _thread
from queue import Queue
from time import sleep

import flappy
from bird import Bird


def create_first_population(population_length=8):
    birds = []
    for _ in range(population_length):
        birds.append(Bird())

    return birds


def sort_birds_by_fitness(birds):
    return sorted(birds, key=lambda x: x.fitness, reverse=True)


def evolve_population(population):
    new_weights = []

    del population[:-1]
    population.append(Bird())

    new_weights.append(Bird.crossover(population[0], population[1]))
    new_weights.append(Bird.crossover(population[1], population[0]))

    for i in range(2, len(population)):
        new_weights.append(population[i])

    for i in range(len(population)):
        new_weights[i] = Bird.mutate(population[i])

        # Update weights
        population[i].model.set_weights(new_weights[i])

    return population


def start_flappy():
    flappy.main()


def observe_bird(bird: Bird, queue: Queue):
    while flappy.is_alive:
        bird.increase_fiteness()
        prediction = bird.should_flap(flappy.diff_x, flappy.diff_y)
        if prediction:
            flappy.flap()
        sleep(0.05)

    queue.put(bird)


def main():
    _thread.start_new_thread(start_flappy, ())
    birds = create_first_population(20)

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

            #_thread.start_new_thread(observe_bird, (bird, queue))
            #bird = queue.get()
            flappy.is_alive = True
            while flappy.is_alive:
                bird.increase_fiteness()
                prediction = bird.should_flap(flappy.diff_x, flappy.diff_y)
                if prediction:
                    flappy.flap()
                sleep(0.05)


            # Updates array
            birds[i] = bird

            print("Generation {}: - Individual {}: - Fitness: {}".format(generation, i, bird.fitness))
            sleep(1)

        birds = sort_birds_by_fitness(birds)
        evolve_population(birds)

        generation += 1


if __name__ == "__main__":
    # todo : set weights
    # print("bonjour")
    main()
