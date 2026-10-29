import os
import sys
import pandas as pd
from src.orchestrator import Plotting

def run_post_processing():
    """
    Standalone runner to aggregate JSON logs, generate analysis plots,
    and export summary tables (CSV and LaTeX) without re-running training.
    """
    print("=== Starting Post-Processing Pipeline ===")
    
    # Initialize the plotting module with output directories
    plotter = Plotting(raw_data_path="outputs/raw_data", plotpath="outputs/plots")
    
    # 1. Extract data from existing JSON result files into a Pandas DataFrame
    print("[1/3] Extracting data from JSON files...")
    plotter.extract_data()

    if not hasattr(plotter, 'df') or plotter.df is None or plotter.df.empty:
        print("❌ Error: No experiment data found in 'outputs/raw_data/'. Exiting.")
        return

    # Identify all unique algorithms present in the extracted data
    detected_algorithms = list(plotter.df['algorithm'].unique())
    print(f"Detected algorithms: {detected_algorithms}")

    # 2. Generate and save the visual plot suite
    print("[2/3] Generating performance plots...")
    try:
        plotter.plot_results(*detected_algorithms)
        print("✅ Visual plots generated successfully in 'outputs/plots/'.")
    except Exception as e:
        print(f"⚠️ Warning: An error occurred while generating plots: {e}")

    # 3. Generate and export the summary tables (CSV & LaTeX)
    print("[3/3] Exporting summary tables...")
    try:
        summary_records = []

        for alg in detected_algorithms:
            df_alg = plotter.df[plotter.df['algorithm'] == alg]
            
            # Locate the index of the highest accuracy using the correct key 'best_accuracy'
            best_idx = df_alg['best_accuracy'].idxmax()
            best_row = df_alg.loc[best_idx]
            
            record = {
                "Algorithm": alg,
                "Best Accuracy (%)": round(best_row['best_accuracy'], 4),
                "Time to Best (s)": round(best_row['time_passed'], 2),
                "Optimal LR": f"{best_row['learning_rate']:.6f}",
                "Optimal Batch Size": int(best_row['batch_size']),
                "Optimal Dropout": f"{best_row['dropout_rate']:.4f}"
            }
            summary_records.append(record)

        summary_df = pd.DataFrame(summary_records).set_index("Algorithm")
        
        # Export using Orchestrator's internal LaTeX/CSV exporter method
        plotter._export_dataframe(
            df=summary_df,
            filename_prefix="final_campaign_summary",
            caption="Summary of optimal performance and hyperparameters found by Swarm Intelligence algorithms.",
            label="tab:performance_summary"
        )
        print("✅ Summary tables successfully exported to 'outputs/raw_data/'.")
        
    except Exception as e:
        print(f"❌ Error generating summary tables: {e}")

    print("=== Post-Processing Completed ===")

if __name__ == "__main__":
    run_post_processing()