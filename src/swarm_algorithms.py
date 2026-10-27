import yaml, os, torch
import numpy as np
from torch import cuda
import concurrent.futures
from tqdm import tqdm


class Particle:
    def __init__(self, position_0, velocity_0, fitness_0):
        self.position = position_0
        self.velocity = velocity_0
        self.fitness = fitness_0
        self.best_position = position_0.copy()
        self.best_fitness = fitness_0

class DungBeetle(Particle):
    def __init__(self, position_0, fitness_0, type='rolling'):
        super().__init__(position_0, None, fitness_0)
        self.type = type  # 'rolling', 'brooding', 'foraging' or 'stealing'
        self.previous_position = position_0.copy()  # Store the previous position for brooding behavior

class SwarmAlgorithm:
    def __init__(self, device, environment=None, total_particel_position_history=False):
        self.environment = environment
        self.device = torch.device(device)
        with open("config.yml", "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        self.safe_vram_margin_mb = config["swarm_optimization"]["safe_vram_margin_mb"]

        # A arbitrary initial value for single model memory usage, will be updated during the first run if auto_calibrate_memory is True
        self.single_model_memory = 600.0 # MB

        # Initialize global best position and fitness
        self.global_best_position = None
        self.global_best_fitness = float('inf') if self.environment.problem_type == 'minimization' else float('-inf')

        # Logging
        self.global_best_fitness_history = []
        self.global_best_position_history = []

        # Initialize particle position history if required
        self.total_particel_position_history = total_particel_position_history
        if self.total_particel_position_history:
            self.particle_position_history = []  # Store the history of all particles' positions

        # Initialize iteration counter
        self.t = 0  # Current iteration

    def initialize_swarm(self, num_particles=20, num_parameters=1, num_iterations=100, auto_calibrate_memory=True):
        self.num_particles = num_particles
        self.num_parameters = num_parameters
        self.num_iterations = num_iterations

        self.particles = []

        if self.device.type == "cuda":
            print(f"\nInitialize swarm with {self.num_particles} particles on GPU in parallel...", flush=True)
            # Compute the maximum number of parallel workers based on available VRAM and estimated memory usage per model
            B_TO_MB = 1024 ** 2
            if auto_calibrate_memory:
                print("Auto-calibrating memory usage for a single model...", flush=True)
                dummy_params = np.random.uniform(0, 1, self.num_parameters)
                self.single_model_memory = self.max_vram_usage(dummy_params) / B_TO_MB

            usable_vram = cuda.get_device_properties(self.device).total_memory / B_TO_MB - self.safe_vram_margin_mb
            max_parallel_workers = min(4, max(1, int(usable_vram // self.single_model_memory)))

            def worker_init():
                os.environ["ARROW_LOG_LEVEL"] = "FATAL"
                os.environ["ARROW_IGNORE_TIMEZONE"] = "1"
                return self._generate_single_particle()

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_workers) as executor:
                # Submit tasks to generate particles in parallel
                futures = [executor.submit(worker_init) for _ in range(self.num_particles)]
                
                # as_completed updates the progress bar as each particle is generated
                for future in tqdm(concurrent.futures.as_completed(futures), total=self.num_particles, desc="Initial Setup (GPU)"):
                    self.particles.append(future.result())
        else:
            print(f"\nInitialize swarm with {self.num_particles} particles on CPU in parallel...", flush=True)
            for _ in tqdm(range(self.num_particles), desc="Initial Setup (CPU)"):
                self.particles.append(self._generate_single_particle())

        # Assuming bounds are [0, 1] for each parameter
        self.bounds = np.array([[0, 1]] * self.num_parameters)

        if self.total_particel_position_history:
            self.particle_position_history.append([particle.position.copy() for particle in self.particles])

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
        self.t += 1  # Increment iteration count

        for particle in self.particles:
            self.global_best_position, self.global_best_fitness = self._update_global_best(particle)
        self.global_best_position_history.append(self.global_best_position)
        self.global_best_fitness_history.append(self.global_best_fitness)

        for particle in self.particles:
            self._update_particle(particle)

        if self.total_particel_position_history:
            self.particle_position_history.append([particle.position.copy() for particle in self.particles])

    def update_particles_gpu(self, auto_calibrate_memory=False):
        self.t += 1  # Increment iteration count

        for particle in self.particles:
            self.global_best_position, self.global_best_fitness = self._update_global_best(particle)
        self.global_best_position_history.append(self.global_best_position)
        self.global_best_fitness_history.append(self.global_best_fitness)

        B_TO_MB = 1024 ** 2  # Bytes to Megabytes conversion factor
        usable_vram = cuda.get_device_properties(self.device).total_memory / B_TO_MB - self.safe_vram_margin_mb
        max_parallel_workers = min(4, max(1, usable_vram // self.single_model_memory))

        def worker_init():
                import os
                os.environ["ARROW_LOG_LEVEL"] = "FATAL"
                os.environ["ARROW_IGNORE_TIMEZONE"] = "1"
                return self._generate_single_particle()

        # Use ThreadPoolExecutor to update particles in parallel, respecting the VRAM limit
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_workers) as executor:
            # Submit tasks to update particles in parallel
            futures = [executor.submit(worker_init) for _ in range(self.num_particles)]
            concurrent.futures.wait(futures) # Wait for all threads to complete

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
            self.global_best_position = particle.position.copy()
            self.global_best_fitness = particle.fitness
        return self.global_best_position, self.global_best_fitness

    def _update_particle(self, particle):
        # This method should be implemented in subclasses (e.g., PSO, DBO) 
        # to define how each particle updates its position and velocity based on the specific algorithm's rules.
        pass

    def run(self, hook=None):
        for iteration in tqdm(range(self.num_iterations), desc="Progresso Globale Sciame"):
            if self.device.type == "cuda":
                cuda.empty_cache()  # Clear GPU memory before starting the optimization
                self.update_particles_gpu()
            else:
                self.update_particles()

            if hook is not None:
                hook()

        if self.total_particel_position_history:
            return self.global_best_position_history, self.global_best_fitness_history, self.particle_position_history
        return self.global_best_position_history, self.global_best_fitness_history

    def reset_swarm(self):
        # Reset the clock for the swarm algorithm
        self.t = 0
        
        # Empty the history of global best positions and fitness values
        self.global_best_fitness_history = []
        self.global_best_position_history = []
        if self.total_particel_position_history:
            self.particle_position_history = []

        # Reset the global best position and fitness
        self.global_best_position = None
        self.global_best_fitness = float('inf') if self.environment.problem_type == 'minimization' else float('-inf')

        # Regenerating new particles
        self.particles = [self._generate_single_particle() for _ in range(self.num_particles)]
        
        if self.total_particel_position_history:
            self.particle_position_history.append([particle.position.copy() for particle in self.particles])

    def max_vram_usage(self, dummy_params):
        """
        Estimate the VRAM usage of a single training of the network

        Args:
            dummy_params (list): A list of dummy parameters to use for the memory calculation.

        Returns:
            float: The estimated VRAM usage in bytes.
        """
        cuda.empty_cache()  # Clear GPU memory before calculating the maximum number of workers
        base_memory = cuda.memory_allocated()
        self._evaluate_fitness(dummy_params)
        max_memory = cuda.max_memory_allocated()
        self.single_model_memory = max_memory - base_memory
        return self.single_model_memory

    def export_state(self):
        """
        Exports the core state common to all swarm algorithms.
        """
        return {
            "positions": self.positions,
            "global_best_position": self.global_best_position,
            "global_best_fitness": self.global_best_fitness,
            "current_iteration": getattr(self, 'current_iteration', 0)
        }

    def import_state(self, state_dict):
        """
        Restores the core state common to all swarm algorithms.
        """
        self.positions = state_dict["positions"]
        self.global_best_position = state_dict["global_best_position"]
        self.global_best_fitness = state_dict["global_best_fitness"]
        self.current_iteration = state_dict["current_iteration"]


class PSO(SwarmAlgorithm):
    def __init__(self, device, environment=None, total_particel_position_history=False):
        # Initialize PSO parameters
        super().__init__(device, environment, total_particel_position_history)

        # PSO hyperparameters
        self.inertial_weight = 0.5
        self.cognitive_weight = 1.5
        self.social_weight = 1.5

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

    def export_state(self):
        """
        Extends the base export_state with PSO-specific kinetic properties.
        """
        state = super().export_state()
        state.update({
            "velocities": self.velocities,
            "personal_best_positions": self.personal_best_positions,
            "personal_best_fitness": self.personal_best_fitness
        })
        return state

    def import_state(self, state_dict):
        """
        Restores base state and PSO-specific properties.
        """
        super().import_state(state_dict)
        self.velocities = state_dict["velocities"]
        self.personal_best_positions = state_dict["personal_best_positions"]
        self.personal_best_fitness = state_dict["personal_best_fitness"]


class DBO(SwarmAlgorithm):
    def __init__(self, device, environment=None, total_particel_position_history=False):
        super().__init__(device, environment, total_particel_position_history)
        self.global_worst_position = None
        self.global_worst_fitness = float('inf') if self.environment.problem_type == 'maximization' else float('-inf')

    def initialize_swarm(self, num_particles=20, num_parameters=3, num_iterations=100, auto_calibrate_memory=True):
        # Probability of different types
        self.type_probabilities = {
            'rolling': 0.2,
            'brooding': 0.2,
            'foraging': 0.4,
            'stealing': 0.2
        }
        super().initialize_swarm(num_particles, num_parameters, num_iterations, auto_calibrate_memory)

    def _generate_single_particle(self):
        dummy_particle = super()._generate_single_particle()
        position = dummy_particle.position
        fitness = dummy_particle.fitness
        particle_type = np.random.choice(list(self.type_probabilities.keys()), p=list(self.type_probabilities.values()))
        return DungBeetle(position, fitness, particle_type)

    def _update_particle(self, particle):
        # Implement brooding behavior update logic
        decay_factor = 1 - (self.t / self.num_iterations)  # Example decay factor based on iteration 
        self.Lb_star = np.maximum(self.global_best_position*(1 - decay_factor), 0.0)
        self.Ub_star = np.minimum(self.global_best_position*(1 + decay_factor), 1.0)
        self.type_functions = {
            'rolling': self._rolling_update,
            'brooding': self._brooding_update,
            'foraging': self._foraging_update,
            'stealing': self._stealing_update
        }
        self.type_functions[particle.type](particle)

        particle.position = np.clip(particle.position, self.bounds[:, 0], self.bounds[:, 1])
        particle.fitness = self._evaluate_fitness(particle.position)

        if self._is_better(particle.fitness, particle.best_fitness):
            particle.best_position = particle.position.copy()
            particle.best_fitness = particle.fitness

    def _update_global_worst(self, particle):
        if not self._is_better(particle.fitness, self.global_worst_fitness):
            self.global_worst_position = particle.position.copy()
            self.global_worst_fitness = particle.fitness

    def update_particles(self):
        # additional logic to update the global worst position and fitness
        for particle in self.particles:
            self._update_global_worst(particle)
        # Call the parent class's update_particles method to handle the rest of the update process
        super().update_particles()

    def update_particles_gpu(self):
        # additional logic to update the global worst position and fitness
        for particle in self.particles:
            self._update_global_worst(particle)
        # Call the parent class's update_particles method to handle the rest of the update process
        super().update_particles_gpu()

    def _rolling_update(self, particle):
        # Implement rolling behavior update logic
        alpha = np.random.choice([-1, 1], size=self.num_parameters)  # Randomly choose direction
        k = np.random.uniform(0, 0.2, size=self.num_parameters)  # Random step size
        b = 0.3 # wind/light constant
        temp_position = particle.previous_position.copy()  # Store the current position before updating
        particle.previous_position = particle.position.copy()  # Store the current position before updating
        delta_x = particle.position - self.global_worst_position
        particle.position += alpha * k * temp_position + b * delta_x

    def _brooding_update(self, particle):
        b1 = np.random.rand(self.num_parameters)
        b2 = np.random.rand(self.num_parameters)
        particle.position = self.global_best_position + b1 * (particle.position - self.Lb_star) + b2 * (particle.position - self.Ub_star)

    def _foraging_update(self, particle):
        # Implement foraging behavior update logic
        c1 = np.random.normal(0, 0.1, size=self.num_parameters)  # Random coefficient
        c2 = np.random.normal(0, 0.1, size=self.num_parameters)  # Random coefficient
        particle.position += c1 * (particle.position - self.Lb_star) + c2 * (particle.position - self.Ub_star)

    def _stealing_update(self, particle):
        # Implement stealing behavior update logic
        S = 0.5  # Stealing factor
        g = np.random.randn(self.num_parameters)  # Random factor
        particle.position = self.global_best_position + S * g * np.abs(self.global_best_position - particle.position)

    def reset_swarm(self):
        super().reset_swarm()
        self.global_worst_position = None
        self.global_worst_fitness = float('inf') if self.environment.problem_type == 'maximization' else float('-inf')

    def export_state(self):
        """
        Extends the base export_state with DBO-specific mathematical properties.
        """
        # 1. Prendi lo stato base dal padre
        state = super().export_state()
        
        # 2. Aggiungi le variabili esclusive del DBO
        state.update({
            "global_worst_position": self.global_worst_position,
            "global_worst_fitness": self.global_worst_fitness
        })
        return state

    def import_state(self, state_dict):
        """
        Restores base state and DBO-specific properties.
        """
        # 1. Fai ripristinare le basi al padre
        super().import_state(state_dict)
        
        # 2. Ripristina le variabili esclusive del DBO
        self.global_worst_position = state_dict["global_worst_position"]
        self.global_worst_fitness = state_dict["global_worst_fitness"]
