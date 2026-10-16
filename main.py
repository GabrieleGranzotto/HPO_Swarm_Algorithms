import time, random
import numpy as np
import torch

from src.orchestrator import Orchestrator, Plotting
from src.test_suite import send_push_notification

def set_global_seed(seed_value: int):
    """
    Enforces deterministic behavior across all mathematical and hardware engines 
    to guarantee exact reproducibility for each independent run.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    TOPIC_NAME = "Project_Swarm_Algorithms_Units"
    algorithms = ["PSO", "DBO"]
    num_runs = 5
    
    time_campaign_start = time.perf_counter()
    
    # 1. Experimental Campaign Execution
    for alg in algorithms:
        for run_idx in range(num_runs):
            message = f"=== Starting {alg} - Run {run_idx + 1}/{num_runs} ==="
            send_push_notification(TOPIC_NAME, "Execution Started", message)
            print(message)
            
            # Re-instantiating the Orchestrator natively resets the internal state
            orchestrator = Orchestrator(full_log=True)
            
            # Dynamic seed assignment ensures statistical diversity between runs 
            # while maintaining strict computational reproducibility
            set_global_seed(42 + run_idx)
            
            orchestrator.run(algorithm=alg)
            
    # 2. Automated Aggregation and Reporting
    time_campaign_end = time.perf_counter()
    total_hours = (time_campaign_end - time_campaign_start) / 3600

    message = f"✅ Experimental Campaign Completed in {total_hours:.2f} hours."

    send_push_notification(TOPIC_NAME, "Experimental Campaign Completed", message)

    print("\n" + "="*80)
    print(message)
    print("📊 Generating automated plots and analytical summaries...")
    print("="*80)
    
    plotter = Plotting()
    plotter.extract_data()
    
    # Generate visual suite and export academic tables (CSV/LaTeX)
    plotter.plot_results(*algorithms)
    plotter.generate_and_export_summary("final_campaign_summary")

    message = f"📊 Automated plots and analytical summaries generated successfully."
    send_push_notification(TOPIC_NAME, "Plots and Summaries Generated", message)