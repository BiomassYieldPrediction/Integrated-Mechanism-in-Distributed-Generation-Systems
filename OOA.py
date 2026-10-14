import time

import numpy as np



def OOA(X, objective_function, VRmin, VRmax, max_iter):
    num_ostriches = X.shape[0]      # Population size
    num_dimensions = X.shape[1]    # Problem dimension
    lower_bound =VRmin[0,:]       # Variable lower limit
    upper_bound =VRmax [0,:]      # Variable upper limit
    fitness = np.array([objective_function(ind) for ind in X])
    best_idx = np.argmin(fitness)
    X_best = X[best_idx].copy()
    Convergence_curve = np.zeros((max_iter))

    t = 0
    ct = time.time()
    for t in range(max_iter):
        for i in range(num_ostriches):
            r1 = np.random.rand()
            r2 = np.random.rand()

            X_new = X[i] + r1 * (X_best - X[i])
            rand_idx = np.random.randint(num_ostriches)
            X_rand = X[rand_idx]
            if r2 < 0.5:
                X_new = X[i] + r2 * (X_rand - X[i])
            else:
                X_new = X[i] - r2 * (X_rand - X[i])

            r3 = np.random.rand()
            X_new = X_new + r3 * (X_best - X_new)

            A = 2 - (2 * t / max_iter)
            X_new = A * X_new

            X_new = np.clip(X_new, lower_bound, upper_bound)

            new_fitness = objective_function(X_new)

            if new_fitness < fitness[i]:
                X[i] = X_new
                fitness[i] = new_fitness
                if new_fitness < objective_function(X_best):
                    X_best = X_new.copy()

        Convergence_curve[t] = new_fitness
        t = t + 1
        Leader_score = Convergence_curve[max_iter - 1]
        ct = time.time() - ct

    return Leader_score, Convergence_curve, X_best, ct
