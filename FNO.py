import numpy as np
import time


def FNO(population, objective_function, lb, ub, max_iter):
    pop_size, dim = population.shape
    fitness = np.array([objective_function(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    convergence = np.zeros(max_iter)
    start_time = time.time()

    for iteration in range(max_iter):
        P_near = iteration / max_iter   # Near probability

        new_population = population.copy()

        for i in range(pop_size):
            r = np.random.rand(dim)

            if np.random.rand() < P_near:
                new_population[i] = population[i] + r * (best_solution - population[i])
            else:
                new_population[i] = population[i] + r * (population[i] - best_solution)

            new_population[i] = np.clip(new_population[i], lb[i], ub[i])

        population = new_population
        fitness = np.array([objective_function(ind) for ind in population])

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_solution = population[best_idx].copy()

        convergence[iteration] = best_fitness

    computation_time = time.time() - start_time
    return best_fitness, convergence, best_solution, computation_time
