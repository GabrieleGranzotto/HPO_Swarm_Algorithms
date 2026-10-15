import random
import numpy as np
import torch

from src.orchestrator import Orchestrator

def set_global_seed(seed_value: int):
    """
    Block the generation of random numbers in Python, NumPy, and PyTorch to ensure reproducibility.
    This function sets the seed for:
    1. Native Python random module
    2. NumPy (crucial for swarm mathematics and matrix operations)
    3. PyTorch CPU (crucial for CNN weights)
    4. Hardware security for GPU acceleration (CUDA)
    """
    # 1. Native Python random module
    random.seed(seed_value)
    
    # 2. NumPy (Matrix and swarm operations)
    np.random.seed(seed_value)
    
    # 3. PyTorch CPU (Weights for CNN)
    torch.manual_seed(seed_value)
    
    # 4. PyTorch GPU (CUDA) - Only if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # multi-GPU configuration
        
        # Force the C++ CUDA algorithms to operate in a strictly deterministic manner
        # (avoid microscopic fluctuations in the CNN convolutional layers)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    algoritmi = ["PSO", "DBO"]
    num_run = 5
    
    for alg in algoritmi:
        for run_idx in range(num_run):
            print(f"=== Starting {alg} - Run {run_idx + 1}/{num_run} ===")
            
            # Re-istantiate the Orchestrator reset natively the data
            orchestrator = Orchestrator(full_log=True)
            
            # the dinamic seed is set to ensure that each run has a unique but reproducible random state
            set_global_seed(42 + run_idx)
            
            orchestrator.run(algorithm=alg)
