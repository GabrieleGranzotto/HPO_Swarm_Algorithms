import numpy as np
from src.pso import PSO

class Funtions:
    @staticmethod
    def sphere(position):
        return sum(x ** 2 for x in position)

    @staticmethod
    def rastrigin(position):
        A = 10
        n = len(position) # dimension of the function
        return A * n + sum(x ** 2 - A * np.cos(2 * np.pi * x) for x in position)

class Benchmark:
    def __init__(self, function, bounds):
        self.function = function
        self.bounds = bounds

    def evaluate(self, position):
        return self.function(position)

    def run(self, method, num_iterations=10):
        best_position = None
        best_fitness = float('inf')

        for _ in range(num_iterations):
            method.update_particles()
            current_best_position = method.global_best_position
            current_best_fitness = method.global_best_fitness

            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_position = current_best_position

        return best_position, best_fitness