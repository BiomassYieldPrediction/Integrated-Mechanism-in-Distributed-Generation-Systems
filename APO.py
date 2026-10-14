import numpy as np
import time


def APO(population, objective_function, lb, ub, max_iter):
    pop_size, dim = population.shape
    alpha = 0.8
    beta = 0.2
    mutation_rate = 0.1
    fitness = np.array([objective_function(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    convergence = np.zeros(max_iter)
    start_time = time.time()

    for iteration in range(max_iter):

        new_population = population.copy()

        for i in range(pop_size):
            r1 = np.random.rand(dim)
            r2 = np.random.rand()
            movement = alpha * r1 * (best_solution - population[i])
            new_population[i] = population[i] + movement
            if r2 < beta:
                new_population[i] = (population[i] + best_solution) / 2
            if np.random.rand() < mutation_rate:
                new_population[i] = lb[i] + np.random.rand(dim) * (ub[i] - lb[i])
            new_population[i] = np.clip(new_population[i], lb[i], ub[i])

        population = new_population
        fitness = np.array([objective_function(ind) for ind in population])

        current_best_idx = np.argmin(fitness)
        current_best_fitness = fitness[current_best_idx]

        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[current_best_idx].copy()

        convergence[iteration] = best_fitness

    computation_time = time.time() - start_time

    return best_fitness, convergence, best_solution, computation_time
