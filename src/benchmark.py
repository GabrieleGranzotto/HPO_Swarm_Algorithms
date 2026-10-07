import numpy as np
import matplotlib.pyplot as plt
from swarm_algorithms import PSO, DBO

class Functions:
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
        self.problem_type = 'minimization'  # Assuming all benchmark functions are minimization problems

    def denormalize_position(self, position):
        bounds_array = np.array(self.bounds)
        low = bounds_array[:, 0]
        high = bounds_array[:, 1]
        return low + position * (high - low)

    def evaluate(self, position):
        # Denormalize the position from [0, 1] to the actual bounds
        return self.function(self.denormalize_position(position))

    def _grid_point_eval(self, num_samples=100):
        x = np.linspace(self.bounds[0][0], self.bounds[0][1], num_samples)
        y = np.linspace(self.bounds[1][0], self.bounds[1][1], num_samples)
        X, Y = np.meshgrid(x, y)
        
        Z = np.zeros_like(X)
        for i in range(num_samples):
            for j in range(num_samples):
                Z[i, j] = self.function(np.array([X[i, j], Y[i, j]]))
                
        return X, Y, Z

class Visualization:
    def __init__(self, benchmark, all_particle_history=None):
        self.benchmark = benchmark
        self.all_particle_history = all_particle_history


    def _plot(self):
        # Create a grid of points
        X, Y, Z = self.benchmark._grid_point_eval()

        # Plotting
        plt.figure(figsize=(10, 8))
        cp = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar(cp)
        plt.title('Benchmark Function Contour Plot')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')

    def _plot_3d(self):
        # Create a grid of points
        X, Y, Z = self.benchmark._grid_point_eval()

        # 3D Plotting
        fig = plt.figure(figsize=(10, 8))
        self.ax = fig.add_subplot(111, projection='3d')
        self.ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.3)
        self.ax.set_title('Benchmark Function 3D Surface Plot')
        self.ax.set_xlabel('X-axis')
        self.ax.set_ylabel('Y-axis')
        self.ax.set_zlabel('Z-axis (Function Value)')

    def plot(self):
        self._plot()
        self._plot_3d()
        plt.show()

    def plot_with_particles(self, positions, save=False, filename='plot_with_particles.png'):
        self._plot()
        # Denormalize particle positions for plotting
        denormalized_positions = np.array([self.benchmark.denormalize_position(position) for position in positions])
        # Plot the particle positions
        
        # Draw the path of the global best position over iterations
        plt.plot(denormalized_positions[:, 0], denormalized_positions[:, 1], 
                 color='red', marker='o', linestyle='-', linewidth=2, label='Global Best Path')
        
        # Mark the final global best position with a star
        plt.scatter(denormalized_positions[-1, 0], denormalized_positions[-1, 1], 
                    color='red', marker='*', s=200, zorder=5, label='Final Solution')

        plt.legend()

        if self.all_particle_history is not None:
            # If we want to plot the history of all particles, we can do so here
            for particle_history in self.all_particle_history:
                denormalized_particle_history = np.array([self.benchmark.denormalize_position(pos) for pos in particle_history])
                plt.plot(denormalized_particle_history[:, 0], denormalized_particle_history[:, 1], 
                         color='black', alpha=0.5, linestyle='--', linewidth=1, label='Particle Path')
                plt.scatter(denormalized_particle_history[-1, 0], denormalized_particle_history[-1, 1], 
                            color='black', marker='*', s=20, zorder=4, label='Final Particle Position')
        
        if save:
            plt.savefig(f"outputs/plots/{filename}")
        plt.show()

    def plot_3d_with_particles(self, positions, save=False, filename='3d_plot_with_particles.png'):
        self._plot_3d()
        # Denormalize particle positions for plotting
        denormalized_positions = np.array([self.benchmark.denormalize_position(position) for position in positions])
        
        # Draw the path of the global best position over iterations
        self.ax.plot(denormalized_positions[:, 0], denormalized_positions[:, 1], 
                [self.benchmark.function(pos) for pos in denormalized_positions], 
                color='red', marker='o', markersize=8, linestyle='-', linewidth=3, label='Global Best Path')
        
        # Mark the final global best position with a star
        final_z = self.benchmark.function(denormalized_positions[-1])
        self.ax.scatter(denormalized_positions[-1, 0], denormalized_positions[-1, 1], final_z, 
                   color='red', marker='*', s=400, zorder=5, label='Final Solution')

        self.ax.legend()

        if self.all_particle_history is not None:
            # If we want to plot the history of all particles, we can do so here
            for particle_history in self.all_particle_history:
                denormalized_particle_history = np.array([self.benchmark.denormalize_position(pos) for pos in particle_history])
                z_values = [self.benchmark.function(pos) for pos in denormalized_particle_history]
                self.ax.plot(denormalized_particle_history[:, 0], denormalized_particle_history[:, 1], 
                        z_values, color='black', alpha=0.5, linestyle='--', linewidth=1, label='Particle Path')
                self.ax.scatter(denormalized_particle_history[-1, 0], denormalized_particle_history[-1, 1], z_values[-1], 
                        color='black', marker='*', s=20, zorder=4, label='Final Particle Position')
        if save:
            plt.savefig(f"outputs/plots/{filename}")
        plt.show()

    def graph_fitness(self, fitness_history, save=False, filename='fitness_history.png'):
        # Set a more visually appealing style
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        iterations = range(len(fitness_history))
        
        # 1. Main line plot for fitness history
        ax.plot(iterations, fitness_history, 
                color='#1f77b4', linewidth=2.5, linestyle='-', 
                marker='o', markersize=6, markerfacecolor='white', markeredgewidth=2, 
                label='Global Best Fitness', zorder=3)
        
        # 2. Fill between the line and the minimum fitness value for a gradient effect
        # This creates a subtle shaded area under the curve to emphasize convergence
        ax.fill_between(iterations, fitness_history, min(fitness_history) * 0.95, 
                        color='#1f77b4', alpha=0.15, zorder=2)

        # 3. Clean up the axes and add grid lines for better readability
        ax.set_title('Convergence Curve - Global Best Fitness', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Iteration', fontsize=12, labelpad=10)
        ax.set_ylabel('Fitness Value (Log/Linear)', fontsize=12, labelpad=10)
        
        # Elegant and lightweight grid lines
        ax.grid(True, linestyle='--', alpha=0.6, zorder=1)
        
        # Add an annotation to the last point (the final result)
        final_iter = iterations[-1]
        final_fit = fitness_history[-1]
        ax.annotate(f'Final: {final_fit:.4e}', 
                    xy=(final_iter, final_fit), 
                    xytext=(final_iter - len(iterations)*0.15, final_fit + (max(fitness_history)-min(fitness_history))*0.1),
                    arrowprops=dict(facecolor='#ff7f0e', shrink=0.05, width=1.5, headwidth=6),
                    fontsize=10, fontweight='bold', color='#333333',
                    bbox=dict(boxstyle="round,pad=0.3", fc="#ff7f0e", alpha=0.2))

        ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
        
        plt.tight_layout()

        if save:
            plt.savefig(f"outputs/plots/{filename}", dpi=300, bbox_inches='tight')
        plt.show()


def benchmark_metric_mesuration():
    bounds = [[-5.12, 5.12], [-5.12, 5.12]] 
    sphere = Benchmark(Functions.sphere, bounds)
    rastrigin = Benchmark(Functions.rastrigin, bounds)
    pso_s = PSO(environment=sphere)
    dbo_s = DBO(environment=sphere)
    pso_r = PSO(environment=rastrigin)
    dbo_r = DBO(environment=rastrigin)
    test_list = [pso_s, dbo_s, pso_r, dbo_r]
    results_test = [[], [], [], []]
    for i in range(4):
        for _ in range(20):
            test_list[i].initialize_swarm(num_particles=20, num_parameters=2, num_iterations=20)
            test_list[i].run()
            results_test[i].append(test_list[i].global_best_fitness)
            test_list[i].reset_swarm()
    for i in range(4):
        class_name = test_list[i].__class__.__name__
        function_name = test_list[i].environment.function.__name__
        best_fitness = np.mean(results_test[i])
        standard_deviation = np.std(results_test[i])
        print(f"Algorithm: {class_name}, Function: {function_name}, Best Fitness: {best_fitness:.3f} ± {standard_deviation:.3f}")
def benchmark_function():
    bounds = [[-5.12, 5.12], [-5.12, 5.12]]
    fun_sphere = Functions.sphere
    fun_rastrigin = Functions.rastrigin
    bench_sphere = Benchmark(fun_sphere, bounds)
    bench_rastrigin = Benchmark(fun_rastrigin, bounds)

    
    # Run PSO
    pso = PSO(environment=bench_sphere, total_particel_position_history=True)
    pso = PSO(environment=bench_rastrigin, total_particel_position_history=True)
    # Run DBO
    dbo = DBO(environment=bench_sphere, total_particel_position_history=True)
    dbo = DBO(environment=bench_rastrigin, total_particel_position_history=True)

    num_particles = 20
    num_parameters = 2
    num_iterations = 20
    pso.initialize_swarm(num_particles, num_parameters, num_iterations)
    positions_pso, fitnesses_pso, all_particle_positions = pso.run()

    dbo.initialize_swarm(num_particles, num_parameters, num_iterations)
    positions_dbo, fitnesses_dbo, all_particle_positions = dbo.run()

    particle_positions_transposed = [list(row) for row in zip(*all_particle_positions)]

    vis = Visualization(bench_sphere, all_particle_history=particle_positions_transposed)
    vis.plot_with_particles(positions_pso, save=True, filename=f'pso_{fun_sphere.__name__}_p{num_particles}_i{num_iterations}_plot_with_particles.png')
    vis.plot_with_particles(positions_dbo, save=True, filename=f'dbo_{fun_sphere.__name__}_p{num_particles}_i{num_iterations}_plot_with_particles.png')
    vis.plot_3d_with_particles(positions_pso, save=True, filename=f'pso_{fun_sphere.__name__}_p{num_particles}_i{num_iterations}_3d_plot_with_particles.png')
    vis.plot_3d_with_particles(positions_dbo, save=True, filename=f'dbo_{fun_sphere.__name__}_p{num_particles}_i{num_iterations}_3d_plot_with_particles.png')
    vis.graph_fitness(fitnesses_pso, save=True, filename=f'pso_{fun_sphere.__name__}_p{num_particles}_i{num_iterations}_fitness_history.png')
    vis.graph_fitness(fitnesses_dbo, save=True, filename=f'dbo_{fun_sphere.__name__}_p{num_particles}_i{num_iterations}_fitness_history.png')
    
    vis = Visualization(bench_rastrigin, all_particle_history=particle_positions_transposed)
    vis.plot_with_particles(positions_pso, save=True, filename=f'pso_{fun_rastrigin.__name__}_p{num_particles}_i{num_iterations}_plot_with_particles.png')
    vis.plot_with_particles(positions_dbo, save=True, filename=f'dbo_{fun_rastrigin.__name__}_p{num_particles}_i{num_iterations}_plot_with_particles.png')
    vis.plot_3d_with_particles(positions_pso, save=True, filename=f'pso_{fun_rastrigin.__name__}_p{num_particles}_i{num_iterations}_3d_plot_with_particles.png')     
    vis.plot_3d_with_particles(positions_dbo, save=True, filename=f'dbo_{fun_rastrigin.__name__}_p{num_particles}_i{num_iterations}_3d_plot_with_particles.png')
    vis.graph_fitness(fitnesses_pso, save=True, filename=f'pso_{fun_rastrigin.__name__}_p{num_particles}_i{num_iterations}_fitness_history.png')
    vis.graph_fitness(fitnesses_dbo, save=True, filename=f'dbo_{fun_rastrigin.__name__}_p{num_particles}_i{num_iterations}_fitness_history.png')

benchmark_function()
benchmark_metric_mesuration()