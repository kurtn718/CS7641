import copy

import numpy as np
import mlrose_hiive as mlrose
import matplotlib.pyplot as plt
import time
from random import randint

# Code in this class is inspired from: https://github.com/soovadeep/cs-7641/blob/master/HW2/flip_flop.py
# I have adapted and modified it to make it into a class,
# so that it can be reused across multiple problems.
#
# It appears that the only thing different in terms of code in these
# type of problems is the Fitness function AND the length, and whether
# we wish to vary any input parameters

class RandomizedProblem:
    def __init__(self,problem_name,fitness):
        self.problem_name = problem_name
        self.fitness = fitness

    def evaluateIncreaseProblemSize(self):
        fitness_sa = []
        fitness_rhc = []
        fitness_ga = []
        fitness_mimic = []
        time_sa = []
        time_rhc = []
        time_ga = []
        time_mimic = []

        range_values = range(5, 145, 20)

        for value in range_values:
            print("Trying range size: ", value)
            fitness = self.fitness
            problem = mlrose.DiscreteOpt(length=value, fitness_fn=self.fitness, maximize=True, max_val=2)
            problem.set_mimic_fast_mode(True)
            init_state = np.random.randint(2, size=value)
            start = time.time()
            _, best_fitness_sa, _ = mlrose.simulated_annealing(problem, schedule=mlrose.ExpDecay(),
                                                                     max_attempts=10, max_iters=2000,
                                                                     init_state=init_state, curve=True)
            end = time.time()
            sa_time = end - start

            start = time.time()
            _, best_fitness_rhc, _ = mlrose.random_hill_climb(problem, max_attempts=10, max_iters=2000,
                                                                    init_state=init_state, curve=True)
            end = time.time()
            rhc_time = end - start

            start = time.time()
            _, best_fitness_ga, _ = mlrose.genetic_alg(problem, max_attempts=10, curve=True)
            end = time.time()
            ga_time = end - start

            start = time.time()
            _, best_fitness_mimic, _ = mlrose.mimic(problem, pop_size=300, max_attempts=10, curve=True)
            end = time.time()
            mimic_time = end - start

            fitness_sa.append(best_fitness_sa)
            fitness_rhc.append(best_fitness_rhc)
            fitness_ga.append(best_fitness_ga)
            fitness_mimic.append(best_fitness_mimic)

            time_sa.append(sa_time)
            time_rhc.append(rhc_time)
            time_ga.append(ga_time)
            time_mimic.append(mimic_time)

        fitness_simulated_annealing = np.array(fitness_sa)
        fitness_random_hill_climb = np.array(fitness_rhc)
        fitness_genetic_algorithm = np.array(fitness_ga)
        fitness_mimic = np.array(fitness_mimic)

        time_simulated_annealing = np.array(time_sa)
        time_random_hill_climb = np.array(time_rhc)
        time_genetic_algorithm = np.array(time_ga)
        time_mimic = np.array(time_mimic)

        plt.figure()
        plt.plot(range_values, fitness_simulated_annealing, label='Simulated Annealing')
        plt.plot(range_values, fitness_random_hill_climb, label='Randomized Hill Climb')
        plt.plot(range_values, fitness_genetic_algorithm, label='Genetic Algorithm')
        plt.plot(range_values, fitness_mimic, label='MIMIC')
        plot_title = 'Fitness vs. Problem Size ' + self.problem_name
        plt.title(plot_title)
        plt.xlabel('Problem Size')
        plt.ylabel('Fitness')
        plt.legend()
        plot_filename = self.problem_name + "_fitness.png"
        plt.savefig(plot_filename)

        plt.figure()
        plt.plot(range_values, time_simulated_annealing, label='Simulated Annealing')
        plt.plot(range_values, time_random_hill_climb, label='Randomized Hill Climb')
        plt.plot(range_values, time_genetic_algorithm, label='Genetic Algorithm')
        plt.plot(range_values, time_mimic, label='MIMIC')
        plot_title = 'Fitness vs. Problem Size ' + self.problem_name
        plt.title(plot_title)
        plt.legend()
        plt.xlabel('Problem Size')
        plt.ylabel('Time (s)')
        plot_filename = self.problem_name + "_time.png"
        plt.savefig(plot_filename)

    def evaluateIncreasingIterations(self,problem_length=150):
        problem = mlrose.DiscreteOpt(length=problem_length, fitness_fn=self.fitness, maximize=True, max_val=2)
        problem.set_mimic_fast_mode(True)
        init_state = np.random.randint(2, size=problem_length)
        _, _, fitness_curve_sa = mlrose.simulated_annealing(problem, schedule=mlrose.ExpDecay(),
                                                                  max_attempts=10, max_iters=2000,
                                                                  init_state=init_state, curve=True)
        print("Complete with SA iterations!")
        _, _, fitness_curve_rhc = mlrose.random_hill_climb(problem, max_attempts=10, max_iters=2000,
                                                                 init_state=init_state, curve=True)
        print("Complete with RHC iterations!")
        _, _, fitness_curve_ga = mlrose.genetic_alg(problem, max_attempts=10, curve=True)
        print("Complete with GA iterations!")
        _, _, fitness_curve_mimic = mlrose.mimic(problem, pop_size=300, max_attempts=10, curve=True)
        print("Complete with MIMIC iterations!")

        plt.figure()
        plt.plot(fitness_curve_sa[:, 0], label='Simulated Annealing')
        plt.plot(fitness_curve_rhc[:, 0], label='Randomized Hill Climb')
        plt.plot(fitness_curve_ga[:, 0], label='Genetic Algorithm')
        plt.plot(fitness_curve_mimic[:, 0], label='MIMIC')
        plot_title = 'Fitness Curve: ' + self.problem_name
        plt.title(plot_title)
        plt.legend()
        plt.xlabel('Iterations')
        plt.ylabel('Fitness')
        plot_filename = self.problem_name + "_iter.png"
        plt.savefig(plot_filename)

    def performAllExperiments(self):
        self.evaluateIncreaseProblemSize()
        self.evaluateIncreasingIterations()

if __name__ == '__main__':
    flip_flop_fitness = mlrose.FlipFlop()
    flip_flop = RandomizedProblem('flip_flop',flip_flop_fitness)
    flip_flop.performAllExperiments()

    fourpeak_fitness = mlrose.FourPeaks(t_pct=0.1)
    fourpeak_problem = RandomizedProblem('four_peaks',fourpeak_fitness)
    fourpeak_problem.performAllExperiments()

    continuous_peak_fitness = mlrose.ContinuousPeaks(t_pct=0.1)
    continuous_peak_problem = RandomizedProblem('continuous_peaks',continuous_peak_fitness)
    continuous_peak_problem.performAllExperiments()

