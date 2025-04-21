#!/usr/bin/env python3
"""
Script to package experiment results and code for publication.
Creates a ZIP file with all necessary files.
"""

import os
import zipfile
import json
import shutil
import datetime
import argparse
from pathlib import Path

def find_latest_experiment():
    """Find the most recent experiment directory"""
    experiments_dir = "experiments"
    all_experiments = [d for d in os.listdir(experiments_dir) if os.path.isdir(os.path.join(experiments_dir, d))]
    all_experiments.sort(reverse=True)  # Sort by timestamp (newest first)
    if not all_experiments:
        raise ValueError("No experiments found.")
    return all_experiments[0]

def create_zip_package(experiment_id=None, output_path=None):
    """Create a ZIP package with experiment results and code"""
    # Find the latest experiment if not specified
    if experiment_id is None:
        experiment_id = find_latest_experiment()
    
    # Set default output path if not provided
    if output_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"phacking_experiment_{experiment_id}_{timestamp}.zip"
    
    print(f"Packaging experiment {experiment_id} to {output_path}...")
    
    # Create the zip file
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add experiment results and metadata
        experiment_dir = os.path.join("experiments", experiment_id)
        
        # Add metadata
        metadata_path = os.path.join(experiment_dir, "metadata.json")
        if os.path.exists(metadata_path):
            zipf.write(metadata_path, f"results/metadata.json")
        
        # Add result files
        results_dir = os.path.join(experiment_dir, "results")
        if os.path.exists(results_dir):
            for file in os.listdir(results_dir):
                if file.endswith('.json'):
                    zipf.write(
                        os.path.join(results_dir, file),
                        f"results/{file}"
                    )
        
        # Add visualizations
        vis_dir = os.path.join(experiment_dir, "visualizations")
        if os.path.exists(vis_dir):
            for file in os.listdir(vis_dir):
                zipf.write(
                    os.path.join(vis_dir, file),
                    f"results/visualizations/{file}"
                )
        
        # Add source code files
        source_files = [
            "run_parallel_experiment.py",
            "analyze_results.py",
            "config/default_experiment.yaml",
        ]
        
        # Add all utility files
        for root, dirs, files in os.walk("utils"):
            for file in files:
                if file.endswith('.py'):
                    source_files.append(os.path.join(root, file))
        
        # Write source files to zip
        for file_path in source_files:
            if os.path.exists(file_path):
                zipf.write(file_path, f"code/{file_path}")
        
        # Create a summary file
        summary_path = "experiment_summary.md"
        with open(summary_path, 'w') as f:
            f.write(f"# P-Hacking Experiment Summary\n\n")
            f.write(f"Experiment ID: {experiment_id}\n")
            
            # Add metadata if available
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as mf:
                    metadata = json.load(mf)
                    f.write(f"Timestamp: {metadata.get('timestamp', 'Unknown')}\n")
                    f.write(f"Analysis model: {metadata.get('model_settings', {}).get('analysis_model', 'Unknown')}\n")
                    f.write(f"Judge model: {metadata.get('model_settings', {}).get('judge_model', 'Unknown')}\n\n")
                    
                    # Add final results
                    final_results = metadata.get("final_results", {})
                    f.write("## Final Results\n\n")
                    f.write(f"Total iterations: {final_results.get('total_iterations', 'Unknown')}\n")
                    f.write(f"Successful iterations: {final_results.get('successful_iterations', 'Unknown')}\n\n")
                    
                    f.write("### Relationship Identification\n")
                    f.write(f"Mean score: {final_results.get('relationship_identification_mean', 'Unknown')}\n")
                    f.write(f"Std score: {final_results.get('relationship_identification_std', 'Unknown')}\n\n")
                    
                    f.write("### Mechanism Explanation\n")
                    f.write(f"Mean score: {final_results.get('mechanism_explanation_mean', 'Unknown')}\n")
                    f.write(f"Std score: {final_results.get('mechanism_explanation_std', 'Unknown')}\n\n")
                    
                    f.write("### P-hacking Detection\n")
                    f.write(f"Rate: {final_results.get('p_hacking_detection_rate', 'Unknown')}\n")
                    f.write(f"Total detected: {final_results.get('p_hacking_total', 'Unknown')}\n\n")
            
            # Add instructions for using the package
            f.write("## Package Contents\n\n")
            f.write("- `results/`: Contains all experiment results and visualizations\n")
            f.write("- `code/`: Contains all source code used for the experiment\n")
            f.write("- `experiment_summary.md`: This summary file\n\n")
            
            f.write("## How to Reproduce\n\n")
            f.write("1. Install required dependencies (see `requirements.txt`)\n")
            f.write("2. Run `python run_parallel_experiment.py` to execute the experiment\n")
            f.write("3. Run `python analyze_results.py` to analyze the results\n")
        
        # Add summary file to zip
        zipf.write(summary_path, "experiment_summary.md")
        
        # Create requirements.txt
        req_path = "requirements.txt"
        with open(req_path, 'w') as f:
            f.write("numpy>=1.20.0\n")
            f.write("pandas>=1.3.0\n")
            f.write("matplotlib>=3.4.0\n")
            f.write("seaborn>=0.11.0\n")
            f.write("tabulate>=0.8.0\n")
            f.write("pyyaml>=6.0\n")
            f.write("scikit-learn>=1.0.0\n")
            f.write("openai>=1.0.0\n")
        
        # Add requirements to zip
        zipf.write(req_path, "code/requirements.txt")
        
        # Clean up temporary files
        os.remove(summary_path)
        os.remove(req_path)
    
    print(f"Package created successfully at {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package experiment results and code for publication")
    parser.add_argument("--experiment", help="Experiment ID to package (defaults to most recent)")
    parser.add_argument("--output", help="Output file path for the ZIP package")
    
    args = parser.parse_args()
    
    try:
        zip_path = create_zip_package(args.experiment, args.output)
        print(f"✅ Successfully created package: {zip_path}")
    except Exception as e:
        print(f"❌ Error creating package: {str(e)}") 