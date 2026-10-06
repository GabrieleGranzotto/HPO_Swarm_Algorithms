import numpy as np

class Particle:
    def __init__(self, position_0, velocity_0, fitness_0):
        self.position = position_0
        self.velocity = velocity_0
        self.fitness = fitness_0
        self.best_position = position_0.copy()
        self.best_fitness = fitness_0

class DungBeetle(Particle):
    def __init__(self, position_0, velocity_0, fitness_0):
        super().__init__(position_0, velocity_0, fitness_0)
        #TODO: Implement Dung Beetle specific attributes and methods
        pass

class SwarmAlgorithm:
    def __init__(self, environment=None, total_particel_position_history=False):
        self.environment = environment
        self.total_particel_position_history = total_particel_position_history

        # Initialize global best position and fitness
        self.global_best_position = None
        self.global_best_fitness = float('inf') if self.environment.problem_type == 'minimization' else float('-inf')

        # Logging
        self.global_best_fitness_history = []
        self.global_best_position_history = []

        if self.total_particel_position_history:
            self.particle_position_history = []  # Store the history of all particles' positions

    def initialize_swarm(self, num_particles=20, num_parameters=1):
        # Initialize Swarm parameters
        self.num_particles = num_particles
        self.num_parameters = num_parameters
        self.particles = [self._generate_single_particle() for _ in range(self.num_particles)]
        # Assuming bounds are [0, 1] for each parameter
        self.bounds = np.array([[0, 1]] * self.num_parameters)

        if self.total_particel_position_history:
            self.particle_position_history.append([particle.position.copy() for particle in self.particles])

    def _generate_single_particle(self):
        pass

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

        if self.total_particel_position_history:
            self.particle_position_history.append([particle.position.copy() for particle in self.particles])

    def _is_better(self, new_fitness, reference_fitness):
        # Determine if the particle's fitness is better than the current global best 
        # based on the problem type (minimization or maximization).
        if self.environment.problem_type == 'minimization':
            return new_fitness < reference_fitness
        else:  # maximization
            return new_fitness > reference_fitness

    def _update_global_best(self, particle):
        if self._is_better(particle.fitness, self.global_best_fitness):
            self.global_best_position = particle.position
            self.global_best_fitness = particle.fitness
        return self.global_best_position, self.global_best_fitness

    def _update_particle(self, particle):
        # This method should be implemented in subclasses (e.g., PSO, DBO) 
        # to define how each particle updates its position and velocity based on the specific algorithm's rules.
        pass

    def run(self, num_iterations=100):
        for iteration in range(num_iterations):
            self.update_particles()
        if self.total_particel_position_history:
            return self.global_best_position_history, self.global_best_fitness_history, self.particle_position_history
        return self.global_best_position_history, self.global_best_fitness_history, self.particle_position_history

class PSO(SwarmAlgorithm):
    def __init__(self, environment=None, total_particel_position_history=False):
        # Initialize PSO parameters
        super().__init__(environment, total_particel_position_history)

        # PSO hyperparameters
        self.inertial_weight = 0.5
        self.cognitive_weight = 1.5
        self.social_weight = 1.5

    def _generate_single_particle(self):
        position = np.random.uniform(0, 1, self.num_parameters)  # Random position in [0, 1]
        velocity = np.random.uniform(-0.1, 0.1, self.num_parameters)  # Random velocity in [-0.1, 0.1]
        fitness = self._evaluate_fitness(position)
        return Particle(position, velocity, fitness)

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
        if self._is_better(particle.fitness, particle.best_fitness):
            particle.best_position = particle.position.copy()
            particle.best_fitness = particle.fitness


class DBO(SwarmAlgorithm):
    def __init__(self, environment=None, total_particel_position_history=False):
        super().__init__(environment, total_particel_position_history)
        #TODO: Implement DBO specific parameters and methods

    def _generate_single_particle(self):
        #TODO: Implement DBO specific particle generation
        pass

    def _update_particle(self, particle):
        #TODO: Implement DBO specific particle update logic
        pass