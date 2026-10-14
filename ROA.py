import time
import numpy as np

def ROA(X, objective_function, VRmin, VRmax, max_iter, rng_seed=None):

    rng = np.random.default_rng(rng_seed)
    num_agents = X.shape[0]
    num_variables = X.shape[1]
    lower_bound = VRmin[0, :]
    upper_bound = VRmax[0, :]
    fitness = np.array([objective_function(ind) for ind in X])
    best_idx = np.argmin(fitness)
    X_best = X[best_idx].copy()
    Leader_score = fitness[best_idx]
    X_prev = X.copy()
    F_prev = fitness.copy()

    Convergence_curve = np.zeros((max_iter, 1))
    start_time = time.time()

    for t in range(max_iter):
        T = max_iter
        weight_leader = (t + 1) / T
        weight_self = 1.0 - weight_leader
        X_p1 = weight_self * X + weight_leader * X_best
        F_p1 = np.array([objective_function(X_p1[i]) for i in range(num_agents)])
        improved = F_p1 < fitness
        X[improved] = X_p1[improved]
        fitness[improved] = F_p1[improved]

        X_p2 = X.copy()
        for i in range(num_agents):
            I = rng.integers(1, 3)
            r = rng.random(num_variables)
            X_p2[i] = X_p2[i] + r * (X_best - I * X_p2[i])
        X_p2 = np.clip(X_p2, lower_bound, upper_bound)
        F_p2 = np.array([objective_function(X_p2[i]) for i in range(num_agents)])
        improved2 = F_p2 < fitness
        X[improved2] = X_p2[improved2]
        fitness[improved2] = F_p2[improved2]

        X_p3 = X.copy()
        for i in range(num_agents):
            r = rng.random(num_variables)
            if F_prev[i] < fitness[i]:
                X_p3[i] = X[i] + r * (X_prev[i] - X[i])
            else:
                X_p3[i] = X[i] + r * (X[i] - X_prev[i])
        X_p3 = np.clip(X_p3, lower_bound, upper_bound)
        F_p3 = np.array([objective_function(X_p3[i]) for i in range(num_agents)])
        improved3 = F_p3 < fitness
        X[improved3] = X_p3[improved3]
        fitness[improved3] = F_p3[improved3]
        X_prev = X.copy()
        F_prev = fitness.copy()

        best_idx = np.argmin(fitness)
        if fitness[best_idx] < Leader_score:
            Leader_score = fitness[best_idx]
            X_best = X[best_idx].copy()

        Convergence_curve[t] = Leader_score

    runtime = time.time() - start_time
    return Leader_score, Convergence_curve, X_best, runtime


