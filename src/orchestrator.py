import yaml, json, time, os, glob, torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from network import NN_Environment
from swarm_algorithms import PSO, DBO

class Orchestrator:
    """
    The class that orchestrate all the project for hyperparameter tuning the CNN problem.
    """
    def __init__(self, config_name:str = "config.yaml", full_log:bool = False):
        self.environment = NN_Environment(device = self.device)
        self.full_log = full_log

        # Load the configuration from the YAML file
        self.config_name = config_name
        with open(self.config_name, "r") as file:
            self.config = yaml.safe_load(file)
        os.makedirs("outputs/logs", exist_ok=True)
        self.logpath = "outputs/logs"
        self.logname = "console_output.log"

        # Determine the device to use for training (GPU, MPS, or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() 
                                   else "mps" if torch.backends.mps.is_available() 
                                   else "cpu")

        # Get the number of iterations and parameters for the optimization algorithms
        self.num_iterations = self.config["swarm_optimization"]["num_iterations"]
        self.num_particles = self.config["swarm_optimization"]["num_particles"]
        self.num_parameters = len(self.config["search_space"])

        # Get the data loaders for training and validation
        self.results_list = []
        self.time_history = []
        self.time_start = 0
        self.time_partial = 0
        self.data_dump_counter = 0
        self.iteration_counter = 0

    def run(self, algorithm: str, force_cpu: bool = False, checkpoint_interval_seconds: int = 900):
        """
        Run the specified optimization algorithm (PSO or DBO) for hyperparameter tuning.
        Includes an automatic checkpointing mechanism to save progress periodically.
        """
        
        # Standard initialization
        if algorithm == "PSO":
            optimizer = PSO(self.environment, self.device)
        elif algorithm == "DBO":
            optimizer = DBO(self.environment, self.device)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        if force_cpu:
            self.environment.device = "cpu"
            optimizer.device = "cpu"

        # --- CHECKPOINT LOGIC: RESTORATION ---
        checkpoint_file = f"{self.logpath}/{algorithm}_checkpoint.pt"
        
        if os.path.exists(checkpoint_file):
            print(f"Checkpoint file found! Restoring {algorithm} state...")
            checkpoint_data = torch.load(checkpoint_file)
            
            # Restore Orchestrator temporal and analytical counters
            self.iteration_counter = checkpoint_data["iteration_counter"]
            self.time_start = time.perf_counter() - checkpoint_data["time_passed_total"]
            self.time_history = checkpoint_data["time_history"]
            self.results_list = checkpoint_data["results_list"]
            
            # Generic swarm initialization, followed by state override
            optimizer.initialize_swarm(self.num_particles, self.num_parameters, self.num_iterations)
            optimizer.import_state(checkpoint_data["optimizer_state"])
            print(f"Resuming from iteration {self.iteration_counter}/{self.num_iterations}")
        else:
            print(f"No checkpoint found. Starting new {algorithm} optimization.")
            optimizer.initialize_swarm(self.num_particles, self.num_parameters, self.num_iterations)
            self.time_start = time.perf_counter()

        # Timer setup for periodic saving
        self.last_save_time = time.perf_counter()

        def take_results():
            self.iteration_counter = getattr(optimizer, 'current_iteration', self.iteration_counter + 1)
            
            [learning_rate, batch_size, dropout_rate] = optimizer.global_best_position
            learning_rate = self.environment._denormalize(learning_rate, "exp", self.environment.lr_log_bounds[0], self.environment.lr_log_bounds[-1])
            dropout_rate = self.environment._denormalize(dropout_rate, "float", self.environment.dropout_bounds[0], self.environment.dropout_bounds[-1])
            batch_size = self.environment._denormalize(batch_size, "int", self.environment.batch_size_choices[0], self.environment.batch_size_choices[-1])

            self.time_partial = time.perf_counter() - self.time_start
            self._print_result_interaction(self.iteration_counter, algorithm, self.time_partial, optimizer.global_best_fitness, learning_rate, batch_size, dropout_rate)
            self._add_experiment_result(algorithm, self.time_partial, optimizer.global_best_fitness, learning_rate, batch_size, dropout_rate)
            self.time_history.append(self.time_partial)
            
            # --- CHECKPOINT LOGIC: SAVING ---
            current_time = time.perf_counter()
            if (current_time - self.last_save_time) >= checkpoint_interval_seconds:
                checkpoint_data = {
                    "optimizer_state": optimizer.export_state(),
                    "iteration_counter": self.iteration_counter,
                    "time_passed_total": self.time_partial,
                    "time_history": self.time_history,
                    "results_list": self.results_list
                }
                # torch.save natively handles PyTorch tensors, NumPy arrays, and dictionaries
                torch.save(checkpoint_data, checkpoint_file)
                self._print_and_log(f"💾 Checkpoint successfully saved at time {self.time_partial:.2f}s")
                self.last_save_time = current_time

        try:
            optimizer.run(hook=take_results)
        finally:
            self._dump_results_to_json()  
            self._archive_json_data()
            
            # --- CHECKPOINT LOGIC: FINAL CLEANUP ---
            # If the required iterations are fully reached, we safely delete the checkpoint file
            if getattr(optimizer, 'current_iteration', self.iteration_counter) >= self.num_iterations:
                try:
                    if os.path.exists(checkpoint_file):
                        os.remove(checkpoint_file)
                        print(f"🧹 Execution completed. Checkpoint file '{os.path.basename(checkpoint_file)}' safely removed.")
                except Exception as e:
                    print(f"Unable to remove checkpoint file: {e}")
                    
        # --- POST-RUN METRICS ---
        self.time_end = time.perf_counter()
        time_passed = self.time_end - self.time_start
        best_learning_rate = self.environment._denormalize(optimizer.global_best_position[0], "exp", self.environment.lr_log_bounds[0], self.environment.lr_log_bounds[-1])
        best_dropout_rate = self.environment._denormalize(optimizer.global_best_position[1], "float", self.environment.dropout_bounds[0], self.environment.dropout_bounds[-1])
        best_batch_size = self.environment._denormalize(optimizer.global_best_position[2], "int", self.environment.batch_size_choices[0], self.environment.batch_size_choices[-1])
        self._print_final_results(algorithm, time_passed, optimizer.global_best_fitness, best_learning_rate, best_batch_size, best_dropout_rate)

    def _print_final_results(self, algorithm, time_passed, best_accuracy, best_learning_rate, best_batch_size, best_dropout_rate):
        """
        Print the final results of the hyperparameter tuning process to the console and log them to a file.
        """
        message = f"""
        =======================================================================
        Final Results of Hyperparameter Tuning:
            - Algorithm: {algorithm}
            - Time Taken: {time_passed:.2f} seconds
            - Iterations: {self.num_iterations}
            - Best Accuracy: {best_accuracy:.4f}
            - Best Hyperparameters:
                - Learning Rate: {best_learning_rate:.6f}
                - Batch Size: {best_batch_size}
                - Dropout Rate: {best_dropout_rate:.2f}
        =======================================================================
        """
        self._print_and_log(message, f"{algorithm}_experiment.log")

    def _print_result_interaction(self, interaction, algorithm, time_passed, accuracy, learning_rate, batch_size, dropout_rate):
        """
        Print the results of an interaction to the console and log it to a file.
        """
        message = f"[{interaction}/{self.num_iterations} - {time_passed:.2f}s] {algorithm} | acc: {accuracy:.4f} | lr: {learning_rate:.6f} | batch: {batch_size} | dropout: {dropout_rate:.2f}"
        self._print_and_log(message, f"{algorithm}_experiment.log")

    def _print_and_log(self, message:str):
        """
        Print a message to the console and log it to a file.

        Args:
            message (str): The message to print and log.
            filename (str): The name of the log file.
            logging (bool, optional): Whether to log the message to a file. Defaults to True.
        """     
        print(message)
        print("-" * 80)
        log_file_path = f"{self.logpath}/{self.logname}"
        if self.full_log:
            try:
                with open(log_file_path, "a") as log_file:
                    timestemp = time.asctime(time.localtime(time.time()))
                    log_file.write(f"[{timestemp}] {message} \n")
            except Exception as e:
                with open(log_file_path, "w") as log_file:
                    log_file.write(f"[{timestemp}] {message} \n")

    def _add_experiment_result(self, algorithm, time_passed, accuracy, learning_rate, batch_size, dropout_rate):
        """
        Collect the results of an experiment and store them in a list. If the number of collected results reaches a certain threshold, dump the results to a JSON file.

        Args:
            algorithm (str): The name of the algorithm used.
            time_passed (float): The time passed for the experiment.
            accuracy (float): The accuracy of the model.
            learning_rate (float): The exponent of the learning rate used.
            batch_size (int): The batch size used.
            dropout_rate (float): The dropout rate used.
        """
        # Create a dictionary to store the experiment result
        experiment_result = {
            "algorithm": algorithm,
            "time_passed": time_passed,
            "best_accuracy": accuracy,
            "best_hyperparameters": {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "dropout_rate": dropout_rate
            }
        }
        # Append the experiment result to the results list
        self.results_list.append(experiment_result)

        self.data_dump_counter += 1
        # If the number of collected results reaches the threshold, dump them to a JSON file
        DUMP_THRESHOLD = 10  # Set the threshold for dumping results
        if self.data_dump_counter >= DUMP_THRESHOLD:
            self._dump_results_to_json()
            self.results_list = []
            self.data_dump_counter = 0

    def _dump_results_to_json(self):
        """
        Dumps the results to a JSON file in the specified directory.

        Args:
            dir (str): The directory where the JSON file will be saved.
        """
        # Open the JSON file in read mode and extract existing data
        try:
            with open(f"{self.logpath}/experiment_results.json", "r") as json_file:
                existing_data = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = []

        # Combine existing data with new data
        combined_data = existing_data + self.results_list

        # Dump the results to a JSON file
        with open(f"{self.logpath}/experiment_results.json", "w") as json_file:
            json.dump(combined_data, json_file, indent=4)

    def _archive_json_data(self):
        """
        Archives the JSON data by renaming the existing file with a timestamp.
        """
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        archive_filename = f"{self.logpath}/experiment_results_{timestamp}.json"
        try:
            os.rename(f"{self.logpath}/experiment_results.json", archive_filename)
            print(f"Archived experiment results to {archive_filename}")
        except FileNotFoundError:
            print("No experiment results to archive.")

class Plotting:
    """
    A class for plotting the results of hyperparameter tuning experiments.
    """
    def __init__(self, logpath="outputs/logs", plotpath="outputs/plots"):
        self.logpath = logpath
        self.plotpath = plotpath

    def _generate_dummy_results(self, *names, num_iterations=10, num_files=2):
        """
        Generates multiple dummy files using the exact same naming convention 
        produced by the Orchestrator, appending a '_dummy' suffix to algorithm names 
        so they can be filtered out easily.
        """
        for n in range(num_files):
            file_results = []
            for name in names:
                for i in range(num_iterations):
                    dummy_result = {
                        "algorithm": name,
                        "time_passed": (i + 1) * 10,  
                        "best_accuracy": min(100.0, (i**1.5) * np.random.uniform(0.8, 1.2) + (n * 3) + 20),  
                        "best_hyperparameters": {
                            "learning_rate": float(10**(-4 + np.random.uniform(0, 3))),
                            "batch_size": int(np.random.choice([16, 32, 64, 128])),
                            "dropout_rate": float(np.random.uniform(0.1, 0.5))
                        }
                    }
                    file_results.append(dummy_result)
            
            # Exactly the same filename structure as the Orchestrator!
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"{self.logpath}/experiment_results_{timestamp}.json"
            
            with open(filename, "w") as json_file:
                json.dump(file_results, json_file, indent=4)
            print(f"Generated test file matching orchestrator format: {os.path.basename(filename)}")
            time.sleep(1) # Ensure unique timestamps

    def _eliminate_dummy_results(self):
        """
        Scans all experiment result files, checks if they contain dummy algorithms, 
        and safely deletes only those files.
        """
        list_of_files = glob.glob(f"{self.logpath}/experiment_results_*.json")
        deleted_count = 0

        for filepath in list_of_files:
            try:
                with open(filepath, "r") as json_file:
                    data = json.load(json_file)
                
                # Check if any record inside the file belongs to a dummy algorithm
                is_dummy_file = any("dummy" in str(row.get("algorithm", "")).lower() for row in data)
                
                if is_dummy_file:
                    os.remove(filepath)
                    print(f"Deleted test file: {os.path.basename(filepath)}")
                    deleted_count += 1
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

        if deleted_count == 0:
            print("No test files found to eliminate.")

    def extract_data(self):
        """
        Reads all JSON log files in the log directory and compiles them into a 
        single Pandas DataFrame. The DataFrame is stored in self.df for unified analysis.
        """
        all_records = []
        
        # Search for all matching JSON files in the log directory
        file_pattern = f"{self.logpath}/experiment_results_*.json"
        list_of_files = glob.glob(file_pattern)
        
        if not list_of_files:
            print("No experiment results found to extract.")
            self.df = pd.DataFrame()  # Initialize empty DataFrame to prevent crashes
            return

        for filepath in list_of_files:
            # Extract timestamp from the filename to use as unique file_id
            # Example filename: "outputs/logs/experiment_results_20261013-130617.json"
            filename = os.path.basename(filepath)
            
            # Remove prefix and extension to isolate the timestamp string
            file_id = filename.replace("experiment_results_", "").replace(".json", "")
            
            try:
                with open(filepath, "r") as json_file:
                    file_results = json.load(json_file)
            except (FileNotFoundError, json.JSONDecodeError):
                print(f"Warning: Could not read or decode {filename}")
                continue
            type_of_algorithms = set(result.get("algorithm") for result in file_results)
            counter_per_algorithm = {type_of_algorithms.pop(): 0 for _ in range(len(type_of_algorithms))}  # Reset counter for each file
            # The JSON contains a list of dictionaries, representing the chronological iterations
            for iteration_idx, result in enumerate(file_results):
                # Flatten the nested structure into a single level dictionary
                counter_per_algorithm[result.get("algorithm")] += 1
                record = {
                    "file_id": file_id,
                    "iteration": counter_per_algorithm[result.get("algorithm")],
                    "algorithm": result.get("algorithm"),
                    "time_passed": result.get("time_passed"),
                    "best_accuracy": result.get("best_accuracy"),
                    
                    # Safely extract hyperparameters even if the key is missing
                    "learning_rate": result.get("best_hyperparameters", {}).get("learning_rate"),
                    "batch_size": result.get("best_hyperparameters", {}).get("batch_size"),
                    "dropout_rate": result.get("best_hyperparameters", {}).get("dropout_rate")
                }
                all_records.append(record)
                
        # Convert the massive list of dictionaries directly into a Pandas DataFrame
        self.df = pd.DataFrame(all_records)
        print(f"Data successfully extracted into self.df with {len(self.df)} total records.")

    def plot_results(self, *algorithms):
        """
        Plots the results of the hyperparameter tuning experiments for the specified algorithm.

        Args:
            algorithms (list): A list of the name of the algorithm used.
        """

        self.plot_accuracy_with_variance(algorithms)
        self.plot_accuracy_spaghetti(algorithms)
        self.plot_parameter_trajectory(algorithms)
        self.plot_final_boxplots(algorithms)
        self.plot_computational_throughput(algorithms)
        self.plot_accuracy_over_time(algorithms)

    def plot_accuracy_with_variance(self, algorithms):
        """
        Plots the average accuracy over iterations with the standard deviation 
        represented as a shaded area. Ideal for academic reports.

        Args:
            algorithms (list): A list of algorithm names to plot (e.g., ["PSO", "DBO"]).
        """
        if self.df is None or self.df.empty:
            print("No data to plot. Please run extract_data() first.")
            return

        # Setup premium aesthetics
        sns.set_theme(style="darkgrid", palette="husl")
        sns.set_context("notebook", rc={"axes.spines.right": False, "axes.spines.top": False})

        plt.figure(figsize=(10, 6))
        plt.title('Average Accuracy with Variance over Iterations', fontsize=16, fontweight='bold')
        
        # Filter the DataFrame to keep only the requested algorithms
        filtered_df = self.df[self.df['algorithm'].isin(algorithms)]

        # Plot the mean line and the standard deviation shadow (errorbar='sd')
        sns.lineplot(data=filtered_df, x='iteration', y='best_accuracy', 
                     hue='algorithm', marker='o', linewidth=2, errorbar='sd')

        plt.xlabel('Iterations', fontsize=12, fontweight='bold')
        plt.ylabel('Best Accuracy (%)', fontsize=12, fontweight='bold')
        plt.legend(title="Algorithm")
        
        # Save the plot dynamically based on the algorithms plotted
        filename_prefix = "_".join(algorithms)
        plt.tight_layout()
        plt.savefig(f"{self.plotpath}/{filename_prefix}_accuracy_variance_plot.png", dpi=300)
        plt.show()

    def plot_accuracy_spaghetti(self, algorithms):
        """
        Plots individual runs as thin semi-transparent lines (spaghetti) 
        and overlays the mean accuracy as a thick solid line. 
        Great for visualizing the step-by-step behavior of the swarm.

        Args:
            algorithms (list): A list of algorithm names to plot (e.g., ["PSO", "DBO"]).
        """
        if self.df is None or self.df.empty:
            print("No data to plot. Please run _extract_data() first.")
            return

        # Setup premium aesthetics
        sns.set_theme(style="darkgrid", palette="husl")
        sns.set_context("notebook", rc={"axes.spines.right": False, "axes.spines.top": False})

        plt.figure(figsize=(10, 6))
        plt.title('Spaghetti Plot: Individual Runs and Mean Accuracy', fontsize=16, fontweight='bold')
        
        # Filter the DataFrame to keep only the requested algorithms
        filtered_df = self.df[self.df['algorithm'].isin(algorithms)]

        # 1. Plot individual runs (Spaghetti background)
        # 'units=file_id' groups the lines by specific run
        # 'estimator=None' prevents Seaborn from calculating the mean here
        # 'alpha=0.3' makes them transparent so they don't overpower the graph
        sns.lineplot(data=filtered_df, x='iteration', y='best_accuracy', 
                     hue='algorithm', units='file_id', estimator=None, 
                     alpha=0.3, linewidth=1, legend=False)

        # 2. Overlay the mean line (Foreground)
        # 'errorbar=None' hides the shadow so we only see the solid average path
        sns.lineplot(data=filtered_df, x='iteration', y='best_accuracy', 
                     hue='algorithm', errorbar=None, linewidth=3.5, marker='o')

        plt.xlabel('Iterations', fontsize=12, fontweight='bold')
        plt.ylabel('Best Accuracy (%)', fontsize=12, fontweight='bold')
        
        # The legend will automatically map the thick lines from step 2
        plt.legend(title="Algorithm Mean")
        
        # Save the plot dynamically
        filename_prefix = "_".join(algorithms)
        plt.tight_layout()
        plt.savefig(f"{self.plotpath}/{filename_prefix}_accuracy_spaghetti_plot.png", dpi=300)
        plt.show()

    def plot_parameter_trajectory(self, algorithms):
        """
        Plots the exploration trajectory in the hyperparameter space (Learning Rate vs Dropout).
        Uses a 'Spaghetti' style to plot all independent runs, showing the clusters of convergence.

        Args:
            algorithms (list): A list of algorithm names to plot (e.g., ["PSO", "DBO"]).
        """
        if self.df is None or self.df.empty:
            print("No data to plot. Please run _extract_data() first.")
            return

        # Setup premium aesthetics
        sns.set_theme(style="darkgrid", palette="husl")
        sns.set_context("notebook", rc={"axes.spines.right": False, "axes.spines.top": False})

        plt.figure(figsize=(10, 8))
        colors = {"PSO": "#3498db", "DBO": "#e67e22"} 
        
        # Filter the DataFrame for the requested algorithms
        filtered_df = self.df[self.df['algorithm'].isin(algorithms)]

        for algorithm in algorithms:
            algo_df = filtered_df[filtered_df['algorithm'] == algorithm]
            if algo_df.empty:
                continue
                
            # Fallback to random color if algorithm is not in the dictionary
            color = colors.get(algorithm, np.random.rand(3,))
            
            # Iterate through each independent run using Pandas groupby
            for file_id, run_df in algo_df.groupby('file_id'):
                # Ensure the data is sorted chronologically by iteration
                run_df = run_df.sort_values('iteration')
                
                lrs = run_df['learning_rate'].values
                dropouts = run_df['dropout_rate'].values
                
                # 1. Trace the spaghetti line (very thin and transparent)
                plt.plot(lrs, dropouts, color=color, alpha=0.2, linewidth=1.0, zorder=1)
                
                # 2. Add scatter points for each step
                plt.scatter(lrs, dropouts, color=color, alpha=0.3, s=20, zorder=2)
                
                # 3. Highlight the Start point (Green square)
                plt.scatter(lrs[0], dropouts[0], color='green', marker='s', s=80, 
                            edgecolor='black', zorder=3, alpha=0.7)
                
                # 4. Highlight the Convergence point (Red star)
                plt.scatter(lrs[-1], dropouts[-1], color='red', marker='*', s=200, 
                            edgecolor='black', zorder=4, alpha=0.9)
            
            # Add a dummy line just to ensure the algorithm name appears cleanly in the legend
            plt.plot([], [], color=color, linewidth=2, label=f"{algorithm} Trajectories")
            
        # Add legend markers for Start and End points (only once)
        plt.scatter([], [], color='green', marker='s', s=80, edgecolor='black', label="Start Point")
        plt.scatter([], [], color='red', marker='*', s=200, edgecolor='black', label="Convergence Point")

        # Log scale is mandatory for Learning Rate
        plt.xscale('log')

        plt.title('Hyperparameter Phase Space: Learning Rate vs Dropout', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Learning Rate (Log Scale)', fontsize=12, fontweight='bold')
        plt.ylabel('Dropout Rate', fontsize=12, fontweight='bold')
        
        # Place legend outside the plot to avoid covering the trajectories
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # Save the plot dynamically
        filename_prefix = "_".join(algorithms)
        plt.savefig(f"{self.plotpath}/{filename_prefix}_trajectory_plot.png", dpi=300)
        plt.show()

    def plot_final_boxplots(self, algorithms):
        """
        Plots a 2x2 grid of boxplots showing the distribution of the FINAL 
        best accuracy and the FINAL chosen hyperparameters across all runs.

        Args:
            algorithms (list): A list of algorithm names to plot (e.g., ["PSO", "DBO"]).
        """
        if self.df is None or self.df.empty:
            print("No data to plot. Please run _extract_data() first.")
            return

        # Setup premium aesthetics
        sns.set_theme(style="darkgrid", palette="husl")
        sns.set_context("notebook", rc={"axes.spines.right": False, "axes.spines.top": False})

        # Filter the DataFrame for the requested algorithms
        filtered_df = self.df[self.df['algorithm'].isin(algorithms)]

        # --- CRITICAL STATISTICAL STEP ---
        # We only want the final results of each run, not the intermediate steps.
        # Find the row index with the maximum 'iteration' for each (algorithm, file_id) pair
        final_idx = filtered_df.groupby(['algorithm', 'file_id'])['iteration'].idxmax()
        # Extract only those final rows
        final_df = filtered_df.loc[final_idx]

        # Create a 2x2 subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Final Optimization Results Distribution (Across All Runs)', 
                     fontsize=18, fontweight='bold', y=0.98)

        # 1. Top-Left: Best Accuracy
        sns.boxplot(data=final_df, x='algorithm', y='best_accuracy', ax=axes[0, 0], palette="husl", hue='algorithm', legend=False)
        # Adding a stripplot overlays the actual data points (stars) on top of the box
        sns.stripplot(data=final_df, x='algorithm', y='best_accuracy', ax=axes[0, 0], 
                      color='black', alpha=0.6, size=6, marker='*')
        axes[0, 0].set_title('Final Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].set_xlabel('')

        # 2. Top-Right: Learning Rate
        sns.boxplot(data=final_df, x='algorithm', y='learning_rate', ax=axes[0, 1], palette="husl", hue='algorithm', legend=False)
        sns.stripplot(data=final_df, x='algorithm', y='learning_rate', ax=axes[0, 1], 
                      color='black', alpha=0.6, size=6, marker='*')
        axes[0, 1].set_title('Final Learning Rate', fontsize=14, fontweight='bold')
        axes[0, 1].set_yscale('log') # Mandatory log scale for LR
        axes[0, 1].set_ylabel('Learning Rate (Log Scale)')
        axes[0, 1].set_xlabel('')

        # 3. Bottom-Left: Batch Size
        sns.boxplot(data=final_df, x='algorithm', y='batch_size', ax=axes[1, 0], palette="husl", hue='algorithm', legend=False)
        sns.stripplot(data=final_df, x='algorithm', y='batch_size', ax=axes[1, 0], 
                      color='black', alpha=0.6, size=6, marker='*')
        axes[1, 0].set_title('Final Batch Size', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Batch Size')
        axes[1, 0].set_xlabel('Algorithm')

        # 4. Bottom-Right: Dropout Rate
        sns.boxplot(data=final_df, x='algorithm', y='dropout_rate', ax=axes[1, 1], palette="husl", hue='algorithm', legend=False)
        sns.stripplot(data=final_df, x='algorithm', y='dropout_rate', ax=axes[1, 1], 
                      color='black', alpha=0.6, size=6, marker='*')
        axes[1, 1].set_title('Final Dropout Rate', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Dropout Rate')
        axes[1, 1].set_xlabel('Algorithm')

        # Adjust layout to prevent overlapping titles/labels
        plt.tight_layout()
        
        # Save the composite image
        filename_prefix = "_".join(algorithms)
        plt.savefig(f"{self.plotpath}/{filename_prefix}_composite_boxplots.png", dpi=300)
        plt.show()

    def plot_computational_throughput(self, algorithms):
        """
        Plots a grouped bar chart comparing the generation time per iteration 
        between algorithms, across multiple runs.
        """
        if self.df is None or self.df.empty:
            print("No data to plot. Please run _extract_data() first.")
            return

        sns.set_theme(style="darkgrid", palette="husl")
        sns.set_context("notebook", rc={"axes.spines.right": False, "axes.spines.top": False})

        # Work on a copy to avoid altering the main database
        filtered_df = self.df[self.df['algorithm'].isin(algorithms)].copy()

        # 1. Strict chronological sorting for each run
        filtered_df = filtered_df.sort_values(['file_id', 'iteration'])
        
        # 2. Compute the Delta Time (time spent exclusively on that iteration)
        filtered_df['generation_time'] = filtered_df.groupby(['algorithm', 'file_id'])['time_passed'].diff()
        
        # 3. Handle the first iteration where delta equals the cumulative time
        filtered_df['generation_time'] = filtered_df['generation_time'].fillna(filtered_df['time_passed'])

        plt.figure(figsize=(14, 6))
        plt.title('Computational Throughput: Time per Iteration Comparison', fontsize=16, fontweight='bold')
        
        # Grouped barplot: x=iteration splits the axis, hue='algorithm' groups side-by-side bars
        sns.barplot(data=filtered_df, x='iteration', y='generation_time', 
                    hue='algorithm', palette="husl", errorbar='sd', capsize=0.1)

        plt.xlabel('Iteration', fontsize=12, fontweight='bold')
        plt.ylabel('Time per Iteration (Seconds)', fontsize=12, fontweight='bold')
        plt.legend(title="Algorithm", loc='upper right')
        
        filename_prefix = "_".join(algorithms)
        plt.tight_layout()
        plt.savefig(f"{self.plotpath}/{filename_prefix}_throughput_grouped_barplot.png", dpi=300)
        plt.show()

    def plot_accuracy_over_time(self, algorithms):
        """
        Plots the evolution of the best accuracy against the cumulative execution time.
        Uses individual run tracing since time intervals are continuous and non-aligned.
        """
        if self.df is None or self.df.empty:
            print("No data to plot. Please run _extract_data() first.")
            return

        sns.set_theme(style="darkgrid", palette="husl")
        sns.set_context("notebook", rc={"axes.spines.right": False, "axes.spines.top": False})

        plt.figure(figsize=(10, 6))
        plt.title('Performance Efficiency: Accuracy vs. Wall-Clock Time', fontsize=16, fontweight='bold')
        
        filtered_df = self.df[self.df['algorithm'].isin(algorithms)]

        sns.lineplot(data=filtered_df, x='time_passed', y='best_accuracy', 
                     hue='algorithm', units='file_id', estimator=None, 
                     marker='o', linewidth=2, alpha=0.7)

        plt.xlabel('Cumulative Execution Time (Seconds)', fontsize=12, fontweight='bold')
        plt.ylabel('Best Accuracy (%)', fontsize=12, fontweight='bold')
        
        plt.legend(title="Algorithm Runs")
        
        filename_prefix = "_".join(algorithms)
        plt.tight_layout()
        plt.savefig(f"{self.plotpath}/{filename_prefix}_accuracy_time_plot.png", dpi=300)
        plt.show()