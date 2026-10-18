import os
# Shut down Arrow logging and timezone warnings to avoid cluttering the output
os.environ["ARROW_LOG_LEVEL"] = "FATAL"
os.environ["ARROW_IGNORE_TIMEZONE"] = "1"
import time, torch, multiprocessing, warnings, requests, sys
# Silence the visible deprecation warnings from NumPy to avoid cluttering the output
warnings.filterwarnings("ignore", message=".*align should be passed.*")

from orchestrator import Orchestrator
from network import NN_Environment
from swarm_algorithms import PSO, DBO

def send_push_notification(topic_name:str, title_message: str, message: str):
    """
    Send a push notification to a specific topic using ntfy.sh service.
    """
    try:
        headers = {"Title": title_message}
        requests.post(f"https://ntfy.sh/{topic_name}", data=message.encode("utf-8"), headers=headers)
    except Exception as e:
        print(f"⚠️ Failed to send push notification: {e}")

class TestingPipeline:
    """
    A class to facilitate testing of the Orchestrator and Plotting classes.
    It runs standard tests to ensure the functionality works as expected.
    It can be run without any supervision and should be totally autonomous.
    """
    def __init__(self):
        os.makedirs("outputs/logs", exist_ok=True)
        self.logpath = "outputs/logs"
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.logname = f"pipeline_test_{timestamp}.log"
        self.flag_test_fail = False

        self.start_test_message = {
            "test_train_nn": "[1/4] Starting test: Network training and testing.",
            "test_small_swarm_cpu": "[2/4] Starting test: Small swarm optimization on CPU.",
            "test_small_swarm_gpu": "[3/4] Starting test: Small swarm optimization on GPU.",
            "test_orchestrator": "[4/4] Starting test: Orchestrator integration."
        }

        self.pass_message = {
            "test_train_nn": "[1/4] Test Passed ✅: Network trained and tested with success.",
            "test_small_swarm_cpu": "[2/4] Test Passed ✅: Small swarm optimization on CPU completed successfully.",
            "test_small_swarm_gpu": "[3/4] Test Passed ✅: Small swarm optimization on GPU completed successfully.",
            "test_orchestrator": "[4/4] Test Passed ✅: Orchestrator integration completed successfully."
        }
        self.fail_message = {
            "test_train_nn": "[1/4] Test Failed ❌: Network training and testing failed.",
            "test_small_swarm_cpu": "[2/4] Test Failed ❌: Small swarm optimization on CPU failed.",
            "test_small_swarm_gpu": "[3/4] Test Failed ❌: Small swarm optimization on GPU failed.",
            "test_orchestrator": "[4/4] Test Failed ❌: Orchestrator integration failed."
        }

    def _print_and_log(self, message: str):
        """
        Print a message to the console and log it to a file.
        """     
        print(message)
        print("-" * 80)
        log_file_path = f"{self.logpath}/{self.logname}"
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        try:
            with open(log_file_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"[{timestamp}] {message} \n")
        except Exception:
            with open(log_file_path, "w", encoding="utf-8") as log_file:
                log_file.write(f"[{timestamp}] {message} \n")

    def _worker_process(self, queue, test_func, *args, **kwargs):
        """
        Executes the test function inside an isolated child process.
        """
        # Silence the verbose logging from Arrow to avoid cluttering the output
        os.environ["ARROW_LOG_LEVEL"] = "ERROR"

        try:
            test_func(*args, **kwargs)
            queue.put(("SUCCESS", None))
        except Exception as e:
            queue.put(("ERROR", str(e)))

    def call_test(self, test_func, *args, **kwargs):
        """
        Wrapper that runs the test in a separate process with a strict 3-hour timeout.
        Handles exceptions, captures hangs, and logs the results.
        """
        timeout = 10800  # 3 hours in seconds (3 * 60 * 60)
        time_start = time.perf_counter()
        
        PROJECT_TOPIC = "Project_Swarm_Algorithms_Units" # Define the topic for push notifications
        
        # Use 'spawn' for safe multithreading and CUDA context isolation
        ctx = multiprocessing.get_context('spawn')
        queue = ctx.Queue()
        
        process = ctx.Process(target=self._worker_process, args=(queue, test_func) + args, kwargs=kwargs)
        process.start()

        # Send push notification for test start
        message = self.start_test_message.get(test_func.__name__, f"Starting test: {test_func.__name__}.")
        self._print_and_log(message)
        
        # Wait for the process to finish or hit the timeout
        process.join(timeout)
        
        if process.is_alive():
            # The test took too long and must be terminated
            process.terminate()
            process.join()
            self.flag_test_fail = True
            message = self.fail_message.get(test_func.__name__, "Test Failed ❌")
            message += f"\nError Details: TimeoutError - The process hung and was forcefully terminated after {timeout} seconds."
        else:
            if not queue.empty():
                status, result = queue.get()
                if status == "ERROR":
                    self.flag_test_fail = True
                    message = self.fail_message.get(test_func.__name__, "Test Failed ❌")
                    message += f"\nError Details: {result}"
                else:
                    message = self.pass_message.get(test_func.__name__, "Test Passed ✅")
            else:
                self.flag_test_fail = True
                message = self.fail_message.get(test_func.__name__, "Test Failed ❌")
                message += "\nError Details: Process terminated unexpectedly at the OS level."
        
        total_time = time.perf_counter() - time_start
        message += f"\nTotal time for this test: {total_time // 3600:.0f} hours {(total_time % 3600) // 60:.0f} minutes {total_time % 60:.2f} seconds."
        self._print_and_log(message)
        # Send push notification for test completion
        send_push_notification(PROJECT_TOPIC, f"Test {test_func.__name__} Completed", message)

    def test_train_nn(self, learning_rate=0.001, batch_size=32, dropout_rate=0.2, num_epochs=5):
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        network = NN_Environment(device = device)
        model = network.model(dropout_rate).to(device)
        train_loader, test_loader = network._get_data_loaders(network.train_dataset, network.test_dataset, batch_size)
        network._train_model(model, train_loader, learning_rate, num_epochs)
        accuracy = network._test_model(model, test_loader)
        if accuracy < 40.0:
            raise ValueError(f"Test accuracy is too low: {accuracy:.2f}%")

    def test_small_swarm_cpu(self, num_particles=5, num_iterations=3):
        device = "cpu"
        network = NN_Environment(device = device)
        algs = [PSO, DBO]
        for alg in algs:
            message = f"Running {alg.__name__} on CPU with small swarm."
            self._print_and_log(message)
            optimizer = alg(network, device)
            optimizer.initialize_swarm(num_particles, 3, num_iterations)
            optimizer.run(hook=lambda: None) 
            if optimizer.global_best_fitness < 20.0:
                raise ValueError(f"{alg.__name__} failed to achieve a reasonable accuracy: {optimizer.global_best_fitness:.2f}")
            message = f"{alg.__name__} on CPU with small swarm completed successfully with accuracy: {optimizer.global_best_fitness:.2f}"
            self._print_and_log(message)

        
    def test_small_swarm_gpu(self, num_particles=5, num_iterations=3):
        if not torch.cuda.is_available():
            raise SystemError("CUDA is not available. GPU test skipped to prevent PyTorch crash.")
        device = "cuda"
        network = NN_Environment(device = device)
        algs = [PSO, DBO]
        for alg in algs:
            message = f"Running {alg.__name__} on GPU with small swarm."
            self._print_and_log(message)
            optimizer = alg(network, device)
            optimizer.initialize_swarm(num_particles, 3, num_iterations, auto_calibrate_memory=True)
            optimizer.run(hook=lambda: None)
            if optimizer.global_best_fitness < 20.0:
                raise ValueError(f"{alg.__name__} failed to achieve a reasonable accuracy on GPU: {optimizer.global_best_fitness:.2f}")
            message = f"{alg.__name__} on GPU with small swarm completed successfully with accuracy: {optimizer.global_best_fitness:.2f}"
            self._print_and_log(message)

    def test_orchestrator(self):
        """
        Test the Orchestrator class by running a small swarm optimization.
        Uses a dedicated testing configuration file.
        """
        algs = ["PSO", "DBO"]
        for alg in algs:
            message = f"Running Orchestrator with {alg.__name__} using the testing configuration."
            self._print_and_log(message)
            orchestrator = Orchestrator(full_log=True)
            # Reintroduced config_name parameter for the custom Orchestrator implementation
            orchestrator.run(algorithm=alg, config_name="config_test.yml")
            message = f"Orchestrator with {alg.__name__} completed successfully."
            self._print_and_log(message)

    def run_all_tests(self, start_idx=1):
        """
        Run all the tests in sequence and log the results.
        """
        message = "="*80 + "\nStarting all tests 🤖. This may take a while depending on your hardware.\n" + "="*80
        self._print_and_log(message)

        tests = [
            self.test_train_nn,
            self.test_small_swarm_cpu,
            self.test_small_swarm_gpu,
            self.test_orchestrator
        ]

        if start_idx < 1 or start_idx > len(tests):
            self._print_and_log(f"⚠️ Errore: L'indice deve essere compreso tra 1 e {len(tests)}.")
            return

        tests_to_run = tests[start_idx - 1:]

        for test in tests_to_run:
            self.call_test(test)
            if self.flag_test_fail:
                self._print_and_log(f"Test {test.__name__} failed. Check logs for details. Pipeline aborted.")
                return  # Stop further tests if one fails
        
        message = "="*80 + "\nAll tests completed without errors 🤖. Please check the logs for detailed results.\n" + "="*80
        self._print_and_log(message)

if __name__ == '__main__':
    start_idx = 1
    if len(sys.argv) > 1:
        try:
            start_idx = int(sys.argv[1])
        except ValueError:
            print(f"⚠️ Invalid argument '{sys.argv[1]}'. Please provide a valid integer index to start from.")
            sys.exit(1)

    # The main block is mandatory when using multiprocessing with 'spawn'
    pipeline = TestingPipeline()
    pipeline.run_all_tests(start_idx=start_idx)

    # Uncomment the following line to test the push notification service independently.
    #send_push_notification("Project_Swarm_Algorithms_Units", "Testing Notification Service", "Testing the push notification service from test_suite.py. If you receive this, the service is working correctly.")