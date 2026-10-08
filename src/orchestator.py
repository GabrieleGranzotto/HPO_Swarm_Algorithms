import yaml
import json
import torch
from src.network import *
from src.swarm_algorithms import PSO, DBO

class Orchestrator:
    """
    The class that orchestrate all the project for hyperparameter tuning the CNN problem.
    """
    def __init__(self):
        self.environment = NN_Environment(device = self.device)

        # Load the configuration from the YAML file
        with open("config.yaml", "r") as file:
            self.config = yaml.safe_load(file)
        self.logpath = "outputs/logs"

        # Determine the device to use for training (GPU, MPS, or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() 
                                   else "mps" if torch.backends.mps.is_available() 
                                   else "cpu")
        self.use_amp = self.config["training_pipeline"]["use_amp"]
        self.safe_vram_margin_mb = self.config["swarm_optimization"]["safe_vram_margin_mb"]

        # Get the number of iterations and parameters for the optimization algorithms
        self.num_iterations = self.config["swarm_optimization"]["num_iterations"]
        self.num_particles = self.config["swarm_optimization"]["num_particles"]
        self.num_parameters = len(self.config["search_space"])

        # Get the data loaders for training and validation
        self.results_list = []
        self.data_dump_counter = 0
        self.iteration_counter = 0

    def run(self, algorithm, safe_vram_margin_mb, use_amp):

        # Initialize the optimizer based on the selected algorithm
        if algorithm == "PSO":
            optimizer = PSO(self.environment)
        elif algorithm == "DBO":
            optimizer = DBO(self.environment)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        optimizer.initialize_swarm(self.num_particles, self.num_parameters, self.num_iterations)
        optimizer.run()

    def _print_final_results(self, time_passed, algorithm, best_accuracy, best_learning_rate, best_batch_size, best_dropout_rate):
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

    def _print_result_interaction(self, interaction, algorithm, accuracy, learning_rate, batch_size, dropout_rate):
        """
        Print the results of an interaction to the console and log it to a file.
        """
        message = f"[{interaction}/{self.num_iterations}] {algorithm} | acc: {accuracy:.4f} | lr: {learning_rate:.6f} | batch: {batch_size} | dropout: {dropout_rate:.2f}"
        self._print_and_log(message, f"{algorithm}_experiment.log")

    def _print_and_log(self, message, filename, logging=True):
        """
        Print a message to the console and log it to a file.

        Args:
            message (str): The message to print and log.
            filename (str): The name of the log file.
            logging (bool, optional): Whether to log the message to a file. Defaults to True.
        """     
        print(message)
        print("-" * 80)
        log_file_path = f"{self.logpath}/{filename}"
        if logging:
            try:
                with open(log_file_path, "a") as log_file:
                    log_file.write(message + "\n")
            except Exception as e:
                with open(log_file_path, "w") as log_file:
                    log_file.write(message + "\n")

    def _add_experiment_result(self, algorithm, accuracy, learning_rate, batch_size, dropout_rate):
        """
        Collect the results of an experiment and store them in a list. If the number of collected results reaches a certain threshold, dump the results to a JSON file.

        Args:
            algorithm (str): The name of the algorithm used.
            accuracy (float): The accuracy of the model.
            learning_rate (float): The exponent of the learning rate used.
            batch_size (int): The batch size used.
            dropout_rate (float): The dropout rate used.
            dir (str, optional): The directory where the JSON file will be saved. Defaults to "outputs/logs".
        """
        # Create a dictionary to store the experiment result
        experiment_result = {
            "algorithm": algorithm,
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
        except FileNotFoundError:
            existing_data = []

        # Combine existing data with new data
        combined_data = existing_data + self.results_list

        # Dump the results to a JSON file
        with open(f"{self.logpath}/experiment_results.json", "w") as json_file:
            json.dump(combined_data, json_file, indent=4)
