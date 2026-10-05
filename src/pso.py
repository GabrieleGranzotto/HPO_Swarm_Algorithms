import numpy as np

class Particle:
    def __init__(self, position_0, velocity_0, fitness_0):
        self.position = position_0
        self.velocity = velocity_0
        self.fitness = fitness_0
        self.best_position = position_0
        self.best_fitness = fitness_0


class PSO:
    def __init__(self, num_particles=20, num_parameters=1, environment=None):
        # Initialize PSO parameters
        self.num_particles = num_particles
        self.particles = list(self._generate_particles(self.num_particles))
        self.num_parameters = num_parameters
        self.environment = environment

        # Initialize global best position and fitness
        self.global_best_position = None
        self.global_best_fitness = float('inf')

        # PSO hyperparameters
        self.inertial_weight = 0.5
        self.cognitive_weight = 1.5
        self.social_weight = 1.5

        self.bounds = np.array([[0, 1]] * self.num_parameters)  # Assuming bounds are [0, 1] for each parameter

        # Logging
        self.global_best_fitness_history = [self.global_best_fitness]
        self.global_best_position_history = [self.global_best_position]

    def _generate_particles(self, num_particles):
        for i in range(num_particles):
            particle = self._generate_single_particle()
            yield particle

    def _generate_single_particle(self):
        position = np.random.uniform(0, 1, self.num_parameters)  # Random position in [0, 1]
        velocity = np.random.uniform(-0.1, 0.1, self.num_parameters)  # Random velocity in [-0.1, 0.1]
        fitness = self._evaluate_fitness(position)
        return Particle(position, velocity, fitness)

    def _evaluate_fitness(self, position):
        if self.environment:
            return self.environment.evaluate(position)
        return 0.0

    def update_particles(self):
        for particle in self.particles:
            self.global_best_position, self.global_best_fitness = self._update_global_best(particle)
        self.global_best_position_history.append(self.global_best_position)
        self.global_best_fitness_history.append(self.global_best_fitness)

        for particle in self.particles:
            self._update_particle(particle)

    def _update_global_best(self, particle):
        if particle.fitness < self.global_best_fitness:
            self.global_best_position = particle.position
            self.global_best_fitness = particle.fitness
        return self.global_best_position, self.global_best_fitness

    def _update_particle(self, particle):
        # Update velocity
        r1 = np.random.rand(self.num_parameters)
        r2 = np.random.rand(self.num_parameters)

        cognitive_component = self.cognitive_weight * r1 * (particle.best_position - particle.position)
        social_component = self.social_weight * r2 * (self.global_best_position - particle.position)

        particle.velocity = (self.inertial_weight * particle.velocity) + cognitive_component + social_component

        # Limit velocity to prevent particles from moving too fast
        MAX_VELOCITY = 0.2
        particle.velocity = np.clip(particle.velocity, -MAX_VELOCITY, MAX_VELOCITY)

        # Update position
        particle.position += particle.velocity

        # Ensure position are within bounds
        particle.position = np.clip(particle.position, self.bounds[:, 0], self.bounds[:, 1])

        # Evaluate fitness
        particle.fitness = self._evaluate_fitness(particle.position)

        # Update personal best if necessary
        if particle.fitness < particle.best_fitness:
            particle.best_position = particle.position
            particle.best_fitness = particle.fitness