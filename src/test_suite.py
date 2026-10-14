import time
import torch
from orchestrator import Orchestrator
from network import NN_Environment
from swarm_algorithms import PSO, DBO

class TestingPipeline:
    """
    A class to facilitate testing of the Orchestrator and Plotting classes.
    It run standard test to ensure the functionality works as expected.
    It can be run without any supervision and should be totally autonomous.
    """
    def __init__(self):
        self.log_path = "outputs/logs"
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.logname = f"pipeline_test_{timestamp}.log"
        self.flag_test_fail = False

        self.pass_message = {
            "test_train_nn": "[1/5] Test Passed ✅: Network trained and tested with success.",
            "test_small_swarm_cpu": "[2/5] Test Passed ✅: Small swarm optimization on CPU completed successfully.",
            "test_big_swarm_cpu": "[3/5] Test Passed ✅: Big swarm optimization on CPU completed successfully.",
            "test_small_swarm_gpu": "[4/5] Test Passed ✅: Small swarm optimization on GPU completed successfully.",
            "test_big_swarm_gpu": "[5/5] Test Passed ✅: Big swarm optimization on GPU completed successfully."
        }
        self.fail_message = {
            "test_train_nn": "[1/5] Test Failed ❌: Network training and testing failed.",
            "test_small_swarm_cpu": "[2/5] Test Failed ❌: Small swarm optimization on CPU failed.",
            "test_big_swarm_cpu": "[3/5] Test Failed ❌: Big swarm optimization on CPU failed.",
            "test_small_swarm_gpu": "[4/5] Test Failed ❌: Small swarm optimization on GPU failed.",
            "test_big_swarm_gpu": "[5/5] Test Failed ❌: Big swarm optimization on GPU failed."
        }

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
        try:
            with open(log_file_path, "a") as log_file:
                timestemp = time.asctime(time.localtime(time.time()))
                log_file.write(f"[{timestemp}] {message} \n")
        except Exception as e:
            with open(log_file_path, "w") as log_file:
                log_file.write(f"[{timestemp}] {message} \n")

    def _time_step_hook(self):
        """
        A hook function to be called at each time step of the optimization process.
        It records the time taken for each iteration and logs it.
        """
        time_partial = time.perf_counter() - self.time_start
        if time_partial > 7_000: # approximately 2 hours in seconds
            raise TimeoutError("The optimization process took too long and was terminated.")

    def call_test(self, test_func, *args, **kwargs):
        """
        Calls a test function and handles exceptions, logging the results.

        Args:
            test_func (callable): The test function to call.
            *args: Positional arguments to pass to the test function.
            **kwargs: Keyword arguments to pass to the test function.
        """
        self.time_start = time.perf_counter()
        try:
            test_func(*args, **kwargs)
            message = self.pass_message[f"{test_func.__name__}"]
        except Exception as e:
            message = self.fail_message[f"{test_func.__name__}"]
            message += f"\nError: {str(e)}"
        total_time = time.perf_counter() - self.time_start
        message += f"\nTotal time for all tests: {total_time//3600:.0f} hours {total_time%3600//60:.0f} minutes {total_time%60:.2f} seconds."
        self._print_and_log(message)

    def test_train_nn(self, learning_rate=0.001, batch_size=32, dropout_rate=0.2, num_epochs=5):
        device = torch.device("cuda" if torch.cuda.is_available() 
                                else "mps" if torch.backends.mps.is_available() 
                                else "cpu")
        network = NN_Environment(device)
        model = network.model(dropout_rate)
        train_loader, test_loader = network._get_data_loaders(network.train_dataset, network.test_dataset, batch_size)
        network._train_model(model, learning_rate, batch_size, dropout_rate, num_epochs)
        accuracy = network._test_model(model, test_loader)
        if accuracy < 40.0:
            raise ValueError(f"Test accuracy is too low: {accuracy:.2f}%")

    def test_small_swarm_cpu(self, num_particles=5, num_iterations=3):
        device = torch.device("cpu")
        network = NN_Environment(device)
        algs = [PSO, DBO]
        for alg in algs:
            optimizer = alg(network, device)
            optimizer.initialize_swarm(num_particles, 3, num_iterations)
            optimizer.run(hook = self._time_step_hook)
            if optimizer.global_best_accuracy < 20.0:
                raise ValueError(f"{alg.__name__} failed to achieve a reasonable accuracy: {optimizer.global_best_accuracy:.2f}%")

    def test_big_swarm_cpu(self, num_particles=20, num_iterations=5):
        device = torch.device("cpu")
        network = NN_Environment(device)
        algs = [PSO, DBO]
        for alg in algs:
            optimizer = alg(network, device)
            optimizer.initialize_swarm(num_particles, 3, num_iterations)
            optimizer.run(hook = self._time_step_hook)
            if optimizer.global_best_accuracy < 20.0:
                raise ValueError(f"{alg.__name__} failed to achieve a reasonable accuracy: {optimizer.global_best_accuracy:.2f}%")
        
    def test_small_swarm_gpu(self, num_particles=5, num_iterations=3):
        device = torch.device("cuda")
        network = NN_Environment(device)
        algs = [PSO, DBO]
        for alg in algs:
            optimizer = alg(network, device)
            optimizer.initialize_swarm(num_particles, 3, num_iterations)
            optimizer.run(hook = self._time_step_hook)
            if optimizer.global_best_accuracy < 20.0:
                raise ValueError(f"{alg.__name__} failed to achieve a reasonable accuracy on GPU: {optimizer.global_best_accuracy:.2f}%")
        
    def test_big_swarm_gpu(self, num_particles=20, num_iterations=5):
        device = torch.device("cuda")
        network = NN_Environment(device)
        algs = [PSO, DBO]
        for alg in algs:
            optimizer = alg(network, device)
            optimizer.initialize_swarm(num_particles, 3, num_iterations)
            optimizer.run(hook = self._time_step_hook)
            if optimizer.global_best_accuracy < 20.0:
                raise ValueError(f"{alg.__name__} failed to achieve a reasonable accuracy on GPU: {optimizer.global_best_accuracy:.2f}%")

    def test_orchestrator(self):
        """
        Test the Orchestrator class by running a small swarm optimization on CPU.
        """
        algs = ["PSO", "DBO"]
        for alg in algs:
            orchestrator = Orchestrator(full_log=True)
            orchestrator.run(alg, config_name="config_test.yml")

    def run_all_tests(self):
        """
        Run all the tests in sequence and log the results.
        """
        message = "="*50 + "\nStarting all tests 🤖. This may take a while depending on your hardware.\n" + "="*50
        self._print_and_log(message)

        tests = [
            self.test_train_nn,
            self.test_small_swarm_cpu,
            self.test_big_swarm_cpu,
            self.test_small_swarm_gpu,
            self.test_big_swarm_gpu
        ]

        for test in tests:
            self.call_test(test)
            if self.flag_test_fail:
                self._print_and_log(f"Test {test.__name__} failed. Check logs for details.")
                return  # Stop further tests if one fails
        
        message = "="*50 + "\nAll tests completed without errors 🤖. Please check the logs for detailed results.\n" + "="*50
        self._print_and_log(message)