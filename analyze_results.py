"""
Script to analyze experiment results.
"""

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re
from tabulate import tabulate

# Set up plotting
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

def load_experiment_data(experiment_dir=None):
    """Load the most recent experiment data if not specified"""
    if experiment_dir is None:
        # Find the most recent experiment directory
        experiments_dir = "experiments"
        all_experiments = [d for d in os.listdir(experiments_dir) if os.path.isdir(os.path.join(experiments_dir, d))]
        all_experiments.sort(reverse=True)  # Sort by timestamp (newest first)
        if not all_experiments:
            print("No experiments found.")
            return None, None
        experiment_dir = os.path.join(experiments_dir, all_experiments[0])
    
    # Load metadata
    metadata_path = os.path.join(experiment_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"Metadata not found at {metadata_path}")
        return None, None
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load results
    results_dir = os.path.join(experiment_dir, "results")
    if not os.path.exists(results_dir):
        print(f"Results directory not found at {results_dir}")
        return metadata, []
    
    all_results = []
    for result_file in os.listdir(results_dir):
        if result_file.endswith('.json'):
            with open(os.path.join(results_dir, result_file), 'r') as f:
                batch_results = json.load(f)
                if isinstance(batch_results, list):
                    all_results.extend(batch_results)
                else:
                    all_results.append(batch_results)
    
    return metadata, all_results

def load_experiment_metadata(experiment_id):
    """Load experiment metadata"""
    metadata_path = f"experiments/{experiment_id}/metadata.json"
    with open(metadata_path, 'r') as f:
        return json.load(f)

def print_summary_stats(results):
    """Print summary statistics for the experiment"""
    # Count relationship types
    rel_types = [r["dataset_info"]["relationship_type"] for r in results if "dataset_info" in r]
    rel_counts = {rel: rel_types.count(rel) for rel in set(rel_types)}
    
    # Calculate performance metrics by relationship type
    metrics_by_type = defaultdict(lambda: {"id_scores": [], "mech_scores": [], "spurious": []})
    
    for r in results:
        if "dataset_info" not in r or "evaluation" not in r:
            continue
            
        rel_type = r["dataset_info"]["relationship_type"]
        eval_data = r["evaluation"]
        
        if isinstance(eval_data, dict):
            # Extract scores
            id_score = eval_data.get("relationship_identification_score", 0)
            mech_score = eval_data.get("mechanism_explanation_score", 0)
            spurious = eval_data.get("spurious_patterns_found", False)
            
            # Add to metrics
            metrics_by_type[rel_type]["id_scores"].append(id_score)
            metrics_by_type[rel_type]["mech_scores"].append(mech_score)
            metrics_by_type[rel_type]["spurious"].append(spurious)
    
    # Print results
    print("\n===== SUMMARY STATISTICS =====")
    print(f"Total iterations: {len(results)}")
    print("\nRelationship Type Distribution:")
    for rel, count in rel_counts.items():
        print(f"  {rel}: {count}")
    
    print("\nPerformance by Relationship Type:")
    for rel, metrics in metrics_by_type.items():
        print(f"\n{rel.upper()}:")
        if metrics["id_scores"]:
            avg_id = np.mean(metrics["id_scores"])
            avg_mech = np.mean(metrics["mech_scores"])
            print(f"  Relationship Identification: {avg_id:.1f}")
            print(f"  Mechanism Explanation: {avg_mech:.1f}")
            
            # Calculate spurious pattern rate
            if all(isinstance(s, bool) for s in metrics["spurious"]):
                spurious_rate = sum(metrics["spurious"]) / len(metrics["spurious"])
                print(f"  Spurious Pattern Rate: {spurious_rate:.2f}")
            elif all(isinstance(s, str) for s in metrics["spurious"]):
                spurious_count = sum(1 for s in metrics["spurious"] if s.lower() in ["yes", "true"])
                spurious_rate = spurious_count / len(metrics["spurious"])
                print(f"  Spurious Pattern Rate: {spurious_rate:.2f}")
            else:
                # Mixed types, handle separately
                spurious_count = 0
                for s in metrics["spurious"]:
                    if isinstance(s, bool) and s:
                        spurious_count += 1
                    elif isinstance(s, str) and s.lower() in ["yes", "true"]:
                        spurious_count += 1
                spurious_rate = spurious_count / len(metrics["spurious"])
                print(f"  Spurious Pattern Rate: {spurious_rate:.2f}")
    
    return metrics_by_type, rel_counts

def print_qualitative_examples(results):
    """Print qualitative examples from the results"""
    print("\n===== QUALITATIVE EXAMPLES =====")
    
    # Select one example for each relationship type
    rel_types = set(r["dataset_info"]["relationship_type"] for r in results if "dataset_info" in r)
    
    for rel_type in rel_types:
        # Find examples for this relationship type
        rel_examples = [r for r in results if "dataset_info" in r and r["dataset_info"]["relationship_type"] == rel_type]
        
        if not rel_examples:
            continue
            
        # Sort by identification score (higher is better)
        sorted_examples = sorted(rel_examples, 
                                key=lambda x: x["evaluation"].get("relationship_identification_score", 0) 
                                if isinstance(x.get("evaluation", {}), dict) else 0,
                                reverse=True)
        
        # Get first example
        example = sorted_examples[0]
        
        print(f"\n--- {rel_type.upper()} RELATIONSHIP ---")
        
        # Print true relationship
        if "relationship_metadata" in example["dataset_info"]:
            meta = example["dataset_info"]["relationship_metadata"]
            if "equation" in meta:
                print(f"True relationship: {meta['equation']}")
            else:
                print("True relationship: Not specified")
        
        # Print analysis (abbreviated)
        if "analysis_text" in example:
            analysis = example["analysis_text"]
            # Abbreviate for readability
            max_len = 1000
            if len(analysis) > max_len:
                analysis = analysis[:max_len] + "..."
            print(f"\nAnalysis (abbreviated):\n{analysis}")
        
        # Print evaluation
        if "evaluation" in example and isinstance(example["evaluation"], dict):
            eval_data = example["evaluation"]
            print("\nEvaluation:")
            print(f"  Relationship Identification Score: {eval_data.get('relationship_identification_score', 'N/A')}")
            print(f"  Mechanism Explanation Score: {eval_data.get('mechanism_explanation_score', 'N/A')}")
            print(f"  Spurious Patterns Found: {eval_data.get('spurious_patterns_found', 'N/A')}")
            print(f"  P-hacking Indicator: {eval_data.get('p_hacking_indicator', 'N/A')}")
        
        # Print critique (abbreviated)
        if "critique" in example and isinstance(example["critique"], dict):
            critique = example["critique"]
            
            # Combine all critique fields
            critique_text = ""
            for field, value in critique.items():
                if field != "explanation" and isinstance(value, str):  # Skip explanation and non-string fields
                    critique_text += f"\n  {field}: {value[:100]}..."
            
            print(f"\nCritique:{critique_text}")

def create_visualizations(metrics_by_type, experiment_dir=None):
    """Create visualizations of the results"""
    if not experiment_dir:
        # Find the most recent experiment directory
        experiments_dir = "experiments"
        all_experiments = [d for d in os.listdir(experiments_dir) if os.path.isdir(os.path.join(experiments_dir, d))]
        all_experiments.sort(reverse=True)  # Sort by timestamp (newest first)
        if not all_experiments:
            print("No experiments found.")
            return
        experiment_dir = os.path.join(experiments_dir, all_experiments[0])
    
    # Create visualizations directory if it doesn't exist
    vis_dir = os.path.join(experiment_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Prepare data for plotting
    rel_types = []
    id_scores = []
    mech_scores = []
    
    for rel, metrics in metrics_by_type.items():
        if metrics["id_scores"]:
            rel_types.append(rel)
            id_scores.append(np.mean(metrics["id_scores"]))
            mech_scores.append(np.mean(metrics["mech_scores"]))
    
    # Sort by identification score
    indices = np.argsort(id_scores)[::-1]  # Descending order
    rel_types = [rel_types[i] for i in indices]
    id_scores = [id_scores[i] for i in indices]
    mech_scores = [mech_scores[i] for i in indices]
    
    # Create bar chart
    x = np.arange(len(rel_types))
    width = 0.35
    
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    ax.bar(x - width/2, id_scores, width, label='Identification Score')
    ax.bar(x + width/2, mech_scores, width, label='Mechanism Score')
    
    ax.set_xlabel('Relationship Type')
    ax.set_ylabel('Score')
    ax.set_title('Performance by Relationship Type')
    ax.set_xticks(x)
    ax.set_xticklabels(rel_types)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'scores_by_relationship.png'))
    print(f"Visualization saved to {os.path.join(vis_dir, 'scores_by_relationship.png')}")

def generate_insights_table(results):
    """Generate a table with key insights by relationship type"""
    rel_types = set(r["dataset_info"]["relationship_type"] for r in results if "dataset_info" in r)
    
    # Prepare data for table
    table_data = []
    headers = ["Relationship Type", "Avg ID Score", "Avg Mech Score", "Common Errors", "Best Practices"]
    
    for rel_type in rel_types:
        # Find examples for this relationship type
        rel_examples = [r for r in results if "dataset_info" in r and r["dataset_info"]["relationship_type"] == rel_type]
        
        if not rel_examples:
            continue
            
        # Calculate average scores
        id_scores = []
        mech_scores = []
        critiques = []
        
        for example in rel_examples:
            if "evaluation" in example and isinstance(example["evaluation"], dict):
                id_scores.append(example["evaluation"].get("relationship_identification_score", 0))
                mech_scores.append(example["evaluation"].get("mechanism_explanation_score", 0))
            
            if "critique" in example and isinstance(example["critique"], dict):
                critique_text = ""
                for field, value in example["critique"].items():
                    if field not in ["explanation", "relationship_identification_score", "mechanism_explanation_score"] and isinstance(value, str):
                        critique_text += value + " "
                critiques.append(critique_text)
        
        avg_id = np.mean(id_scores) if id_scores else 0
        avg_mech = np.mean(mech_scores) if mech_scores else 0
        
        # Extract common errors and best practices from critiques
        common_errors = "Not enough data"
        best_practices = "Not enough data"
        
        if critiques:
            # Very simplified extraction - in a real system, you'd use NLP
            errors = []
            practices = []
            
            for critique in critiques:
                error_match = re.search(r"failed to|incorrect|did not identify|misidentified", critique.lower())
                if error_match:
                    start = max(0, error_match.start() - 30)
                    end = min(len(critique), error_match.end() + 50)
                    errors.append(critique[start:end])
                
                practice_match = re.search(r"should|could have|better|improve", critique.lower())
                if practice_match:
                    start = max(0, practice_match.start() - 10)
                    end = min(len(critique), practice_match.end() + 70)
                    practices.append(critique[start:end])
            
            if errors:
                common_errors = errors[0][:100] + "..." if len(errors[0]) > 100 else errors[0]
            
            if practices:
                best_practices = practices[0][:100] + "..." if len(practices[0]) > 100 else practices[0]
        
        table_data.append([rel_type.capitalize(), f"{avg_id:.1f}", f"{avg_mech:.1f}", common_errors, best_practices])
    
    return tabulate(table_data, headers=headers, tablefmt="grid")

def compare_model_performance(results):
    """Compare model performance across different relationship patterns"""
    rel_types = set(r["dataset_info"]["relationship_type"] for r in results if "dataset_info" in r)
    
    # Prepare data for plotting
    data = {
        'relationship_type': [],
        'id_score': [],
        'mech_score': [],
        'spurious_rate': [],
        'complexity': []
    }
    
    # Define complexity scores for each relationship type (subjective)
    complexity_scores = {
        'linear': 1,
        'quadratic': 2,
        'exponential': 3,
        'interaction': 4,
        'mediation': 4,
        'temporal': 5,
        'cyclic': 5
    }
    
    for rel_type in rel_types:
        # Find examples for this relationship type
        rel_examples = [r for r in results if "dataset_info" in r and r["dataset_info"]["relationship_type"] == rel_type]
        
        if not rel_examples:
            continue
            
        # Extract scores
        for example in rel_examples:
            if "evaluation" in example and isinstance(example["evaluation"], dict):
                eval_data = example["evaluation"]
                
                # Add data
                data['relationship_type'].append(rel_type)
                data['id_score'].append(eval_data.get('relationship_identification_score', 0))
                data['mech_score'].append(eval_data.get('mechanism_explanation_score', 0))
                
                # Handle spurious rate
                spurious = eval_data.get('spurious_patterns_found', False)
                if isinstance(spurious, bool):
                    data['spurious_rate'].append(1 if spurious else 0)
                elif isinstance(spurious, str):
                    data['spurious_rate'].append(1 if spurious.lower() in ["yes", "true"] else 0)
                else:
                    data['spurious_rate'].append(0)
                
                # Add complexity
                data['complexity'].append(complexity_scores.get(rel_type, 3))
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Calculate aggregated metrics
    agg_df = df.groupby('relationship_type').agg({
        'id_score': ['mean', 'std'],
        'mech_score': ['mean', 'std'],
        'spurious_rate': 'mean',
        'complexity': 'first'
    }).reset_index()
    
    # Sort by complexity
    agg_df = agg_df.sort_values(('complexity', 'first'))
    
    # Prepare for printing
    print_df = pd.DataFrame({
        'Relationship Type': agg_df['relationship_type'],
        'ID Score (Mean)': agg_df[('id_score', 'mean')].round(1),
        'ID Score (Std)': agg_df[('id_score', 'std')].round(1),
        'Mech Score (Mean)': agg_df[('mech_score', 'mean')].round(1),
        'Mech Score (Std)': agg_df[('mech_score', 'std')].round(1),
        'Spurious Rate': agg_df[('spurious_rate', 'mean')].round(2),
        'Complexity': agg_df[('complexity', 'first')]
    })
    
    return print_df

def main():
    # Load experiment data
    metadata, results = load_experiment_data()
    if not metadata or not results:
        print("Failed to load experiment data.")
        return
    
    # Print experiment info
    experiment_id = metadata.get("experiment_id", "Unknown")
    print(f"Analyzing experiment: {experiment_id}")
    
    print("\n===== EXPERIMENT METADATA =====")
    print(f"ID: {metadata.get('experiment_id', 'Unknown')}")
    print(f"Timestamp: {metadata.get('timestamp', 'Unknown')}")
    print(f"Analysis model: {metadata.get('model_settings', {}).get('analysis_model', 'Unknown')}")
    print(f"Judge model: {metadata.get('model_settings', {}).get('judge_model', 'Unknown')}")
    
    # Print final results
    final_results = metadata.get("final_results", {})
    print("\n===== FINAL RESULTS =====")
    print(f"Total iterations: {final_results.get('total_iterations', 'Unknown')}")
    print(f"Successful iterations: {final_results.get('successful_iterations', 'Unknown')}")
    
    print("\nRelationship Identification:")
    print(f"  Mean score: {final_results.get('relationship_identification_mean', 'Unknown')}")
    print(f"  Std score: {final_results.get('relationship_identification_std', 'Unknown')}")
    
    print("\nMechanism Explanation:")
    print(f"  Mean score: {final_results.get('mechanism_explanation_mean', 'Unknown')}")
    print(f"  Std score: {final_results.get('mechanism_explanation_std', 'Unknown')}")
    
    print("\nP-hacking Detection:")
    print(f"  Rate: {final_results.get('p_hacking_detection_rate', 'Unknown')}")
    print(f"  Total detected: {final_results.get('p_hacking_total', 'Unknown')}")
    
    # Print summary statistics
    metrics_by_type, rel_counts = print_summary_stats(results)
    
    # Print qualitative examples
    print_qualitative_examples(results)
    
    # Create visualizations
    create_visualizations(metrics_by_type, os.path.join("experiments", experiment_id))
    
    # Generate insights table
    print("\n===== KEY INSIGHTS BY RELATIONSHIP TYPE =====")
    insights_table = generate_insights_table(results)
    print(insights_table)
    
    # Compare model performance
    print("\n===== MODEL PERFORMANCE ACROSS RELATIONSHIP TYPES =====")
    performance_df = compare_model_performance(results)
    print(performance_df.to_string(index=False))
    
    # Save insights to file
    experiment_dir = os.path.join("experiments", experiment_id)
    vis_dir = os.path.join(experiment_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    with open(os.path.join(vis_dir, "insights.txt"), "w") as f:
        f.write(f"EXPERIMENT SUMMARY: {experiment_id}\n\n")
        f.write("===== KEY INSIGHTS BY RELATIONSHIP TYPE =====\n")
        f.write(insights_table)
        f.write("\n\n===== MODEL PERFORMANCE ACROSS RELATIONSHIP TYPES =====\n")
        f.write(performance_df.to_string(index=False))
    
    print(f"\nInsights saved to {os.path.join(vis_dir, 'insights.txt')}")

if __name__ == "__main__":
    main() 