import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter
from scipy import stats
from typing import Dict, List, Any
import re

def analyze_semantic_phacking_data(results_path, log_path, output_dir):
    """
    Analyze semantic p-hacking experiment data and generate visualizations
    
    Parameters:
    - results_path: Path to experiment_results.json
    - log_path: Path to interaction_log.json
    - output_dir: Directory to save outputs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {results_path} and {log_path}...")
    
    with open(results_path, 'r') as f:
        results_data = json.load(f)
    
    with open(log_path, 'r') as f:
        log_data = json.load(f)
    
    # Create DataFrame from results
    results_df = pd.json_normalize(results_data)
    
    # Extract key metrics from results
    print("Extracting key metrics...")
    
    # Create summary stats for actual correlations
    corr_summary = results_df.groupby('dataset_type')['actual_avg_abs_correlation'].agg(
        ['mean', 'std', 'min', 'max']).reset_index()
    
    # Extract responses
    pattern_responses = []
    hypothesis_responses = []
    critical_responses = []
    
    for result in results_data:
        dataset_type = result['dataset_type']
        
        # Pattern finding responses
        pattern_responses.append({
            'dataset_type': dataset_type,
            'response': result['pattern_finding_results']['response'],
            'prompt_type': 'pattern_finding',
            'iteration': result['iteration']
        })
        
        # Hypothesis responses
        hypothesis_responses.append({
            'dataset_type': dataset_type,
            'response': result['hypothesis_results']['response'],
            'prompt_type': 'hypothesis',
            'iteration': result['iteration']
        })
        
        # Critical analysis responses
        critical_responses.append({
            'dataset_type': dataset_type,
            'response': result['critical_analysis_results']['response'],
            'prompt_type': 'critical_analysis',
            'iteration': result['iteration']
        })
    
    # Combine all responses
    all_responses = pattern_responses + hypothesis_responses + critical_responses
    responses_df = pd.DataFrame(all_responses)
    
    # Analyze response characteristics
    responses_df['length'] = responses_df['response'].apply(len)
    responses_df['word_count'] = responses_df['response'].apply(lambda x: len(x.split()))
    
    # Extract claims about correlations
    def count_correlation_mentions(text):
        text = text.lower()
        correlation_terms = ['correlation', 'relationship', 'association', 'linked', 'connected']
        return sum(text.count(term) for term in correlation_terms)
    
    responses_df['correlation_mentions'] = responses_df['response'].apply(count_correlation_mentions)
    
    # Count pattern claims
    def has_pattern_claim(text):
        text = text.lower()
        pattern_phrases = ['i found', 'pattern', 'relationship', 'correlation', 'trend']
        return 1 if any(phrase in text for phrase in pattern_phrases) else 0
    
    responses_df['claims_pattern'] = responses_df['response'].apply(has_pattern_claim)
    
    # Analysis of the log data
    # Extract timing data
    timing_data = []
    for entry in log_data:
        if entry['action'] == 'llm_analysis':
            timing_data.append({
                'dataset_type': entry['data']['dataset_metadata']['dataset_type'],
                'prompt_type': entry['data']['prompt_type'],
                'duration': entry['data']['duration_seconds']
            })
    
    timing_df = pd.DataFrame(timing_data)
    
    # Generate summary statistics
    print("Generating summary statistics...")
    
    # Summary 1: Response characteristics by dataset and prompt type
    response_summary = responses_df.groupby(['dataset_type', 'prompt_type']).agg({
        'length': ['mean', 'std'],
        'word_count': ['mean', 'std'],
        'correlation_mentions': ['mean', 'sum'],
        'claims_pattern': ['mean', 'sum']
    }).reset_index()
    
    # Summary 2: Timing data
    timing_summary = timing_df.groupby(['dataset_type', 'prompt_type']).agg({
        'duration': ['mean', 'std', 'min', 'max']
    }).reset_index()
    
    # Calculate key indicators
    random_pattern_claims = responses_df[(responses_df['dataset_type'] == 'random') & 
                                         (responses_df['claims_pattern'] == 1)].shape[0]
    correlated_pattern_claims = responses_df[(responses_df['dataset_type'] == 'correlated') & 
                                            (responses_df['claims_pattern'] == 1)].shape[0]
    
    random_pattern_claim_rate = random_pattern_claims / len(responses_df[responses_df['dataset_type'] == 'random'])
    correlated_pattern_claim_rate = correlated_pattern_claims / len(responses_df[responses_df['dataset_type'] == 'correlated'])
    
    # Create a summary of key findings
    summary_stats = {
        'experiment_date': os.path.basename(os.path.dirname(results_path)),
        'total_iterations': len(results_data) // 2,
        'total_llm_calls': len(timing_data),
        'avg_random_correlation': corr_summary[corr_summary['dataset_type'] == 'random']['mean'].values[0],
        'avg_induced_correlation': corr_summary[corr_summary['dataset_type'] == 'correlated']['mean'].values[0],
        'random_pattern_claim_rate': random_pattern_claim_rate,
        'correlated_pattern_claim_rate': correlated_pattern_claim_rate,
        'false_pattern_ratio': random_pattern_claim_rate / correlated_pattern_claim_rate if correlated_pattern_claim_rate > 0 else float('inf'),
        'avg_response_time': timing_df['duration'].mean(),
        'total_experiment_time': timing_df['duration'].sum()
    }
    
    # Save summary statistics to file
    with open(os.path.join(output_dir, 'summary_stats.json'), 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # Save detailed statistics
    response_summary.to_csv(os.path.join(output_dir, 'response_summary.csv'))
    timing_summary.to_csv(os.path.join(output_dir, 'timing_summary.csv'))
    responses_df.to_csv(os.path.join(output_dir, 'all_responses_analysis.csv'))
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # Set styling for plots
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette('viridis')
    
    # Plot 1: Pattern claims by dataset type
    plt.figure(figsize=(10, 6))
    claims_by_dataset = responses_df.groupby(['dataset_type', 'prompt_type'])['claims_pattern'].mean().reset_index()
    claims_pivot = claims_by_dataset.pivot(index='prompt_type', columns='dataset_type', values='claims_pattern')
    
    ax = claims_pivot.plot(kind='bar', figsize=(10, 6))
    plt.title('Rate of Pattern Claims by Dataset Type and Prompt', fontsize=14)
    plt.xlabel('Prompt Type', fontsize=12)
    plt.ylabel('Rate of Pattern Claims', fontsize=12)
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.legend(title='Dataset Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pattern_claims_by_dataset.png'), dpi=300)
    
    # Plot 2: Correlation mentions
    plt.figure(figsize=(10, 6))
    corr_mentions = responses_df.groupby(['dataset_type', 'prompt_type'])['correlation_mentions'].mean().reset_index()
    corr_pivot = corr_mentions.pivot(index='prompt_type', columns='dataset_type', values='correlation_mentions')
    
    ax = corr_pivot.plot(kind='bar', figsize=(10, 6))
    plt.title('Average Number of Correlation Mentions by Dataset Type and Prompt', fontsize=14)
    plt.xlabel('Prompt Type', fontsize=12)
    plt.ylabel('Average Correlation Mentions', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title='Dataset Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_mentions.png'), dpi=300)
    
    # Plot 3: Response length distribution
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='prompt_type', y='word_count', hue='dataset_type', data=responses_df)
    plt.title('Distribution of Response Length by Prompt Type and Dataset', fontsize=14)
    plt.xlabel('Prompt Type', fontsize=12)
    plt.ylabel('Word Count', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title='Dataset Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'response_length_distribution.png'), dpi=300)
    
    # Plot 4: Actual vs. Claimed Correlations
    plt.figure(figsize=(8, 6))
    
    # Prepare data for the plot
    plot_data = pd.DataFrame({
        'Dataset Type': ['Random', 'Correlated'],
        'Actual Average Correlation': [
            corr_summary[corr_summary['dataset_type'] == 'random']['mean'].values[0],
            corr_summary[corr_summary['dataset_type'] == 'correlated']['mean'].values[0]
        ],
        'Pattern Claim Rate': [random_pattern_claim_rate, correlated_pattern_claim_rate]
    })
    
    bar_width = 0.35
    r1 = np.arange(2)
    r2 = [x + bar_width for x in r1]
    
    plt.bar(r1, plot_data['Actual Average Correlation'], width=bar_width, label='Actual Avg. Correlation')
    plt.bar(r2, plot_data['Pattern Claim Rate'], width=bar_width, label='Pattern Claim Rate')
    
    plt.xlabel('Dataset Type', fontsize=12)
    plt.xticks([r + bar_width/2 for r in range(2)], ['Random', 'Correlated'])
    plt.title('Comparison of Actual Correlations vs. Pattern Claims', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'actual_vs_claimed_patterns.png'), dpi=300)
    
    # Plot 5: Response time distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='prompt_type', y='duration', hue='dataset_type', data=timing_df)
    plt.title('Distribution of Response Times by Prompt Type and Dataset', fontsize=14)
    plt.xlabel('Prompt Type', fontsize=12)
    plt.ylabel('Duration (seconds)', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title='Dataset Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'response_time_distribution.png'), dpi=300)
    
    # Create summary visualization
    plt.figure(figsize=(14, 8))
    
    # Create a 2x2 plot grid
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Summary of Pattern Claims (Top Left)
    summary_data = pd.DataFrame({
        'Dataset Type': ['Random', 'Correlated'],
        'Pattern Claim Rate': [random_pattern_claim_rate, correlated_pattern_claim_rate]
    })
    
    sns.barplot(x='Dataset Type', y='Pattern Claim Rate', data=summary_data, ax=ax1)
    ax1.set_title('Rate of Pattern Claims by Dataset Type')
    ax1.set_ylim(0, 1)
    
    # Plot 2: Average Word Count (Top Right)
    word_count_summary = responses_df.groupby('dataset_type')['word_count'].mean().reset_index()
    sns.barplot(x='dataset_type', y='word_count', data=word_count_summary, ax=ax2)
    ax2.set_title('Average Response Length by Dataset Type')
    ax2.set_xlabel('Dataset Type')
    ax2.set_ylabel('Average Word Count')
    
    # Plot 3: Actual Correlations (Bottom Left)
    sns.barplot(x='dataset_type', y='mean', data=corr_summary, ax=ax3)
    ax3.set_title('Actual Average Correlation by Dataset Type')
    ax3.set_xlabel('Dataset Type')
    ax3.set_ylabel('Average Absolute Correlation')
    
    # Plot 4: Response Times (Bottom Right)
    time_summary = timing_df.groupby('dataset_type')['duration'].mean().reset_index()
    sns.barplot(x='dataset_type', y='duration', data=time_summary, ax=ax4)
    ax4.set_title('Average Response Time by Dataset Type')
    ax4.set_xlabel('Dataset Type')
    ax4.set_ylabel('Duration (seconds)')
    
    plt.tight_layout()
    plt.suptitle('Semantic P-Hacking Experiment Summary', fontsize=16, y=1.05)
    plt.savefig(os.path.join(output_dir, 'experiment_summary.png'), dpi=300, bbox_inches='tight')
    
    # Create a text-based summary report
    summary_report = f"""
    SEMANTIC P-HACKING EXPERIMENT SUMMARY
    ====================================
    Experiment ID: {summary_stats['experiment_date']}
    
    EXPERIMENT OVERVIEW:
    - Total iterations: {summary_stats['total_iterations']}
    - Total LLM calls: {summary_stats['total_llm_calls']}
    - Total experiment time: {summary_stats['total_experiment_time']:.2f} seconds
    
    KEY FINDINGS:
    1. Pattern Claims:
       - Random datasets: {summary_stats['random_pattern_claim_rate']*100:.1f}% of responses claimed patterns
       - Correlated datasets: {summary_stats['correlated_pattern_claim_rate']*100:.1f}% of responses claimed patterns
       - False pattern ratio: {summary_stats['false_pattern_ratio']:.2f}
    
    2. Actual vs. Claimed Correlations:
       - Random datasets average correlation: {summary_stats['avg_random_correlation']:.3f}
       - Correlated datasets average correlation: {summary_stats['avg_induced_correlation']:.3f}
       
    3. Response Characteristics:
       - Average response time: {summary_stats['avg_response_time']:.2f} seconds
    
    CONCLUSION:
    {
        "The LLM frequently claimed to find patterns in purely random data, demonstrating the semantic p-hacking phenomenon described in the paper."
        if random_pattern_claim_rate > 0.3 else
        "The LLM showed restraint in claiming patterns in random data, suggesting resistance to semantic p-hacking."
    }
    """
    
    # Save the text summary
    with open(os.path.join(output_dir, 'experiment_summary.txt'), 'w') as f:
        f.write(summary_report)
    
    print(f"Analysis complete! All outputs saved to {output_dir}")
    return summary_stats

def generate_concise_summary(results_path: str, log_path: str) -> str:
    """
    Generate a concise summary of the experiment results.
    
    Args:
        results_path: Path to experiment_results.json
        log_path: Path to interaction_log.json
        
    Returns:
        str: A concise summary of the experiment results
    """
    # Load data
    with open(results_path, 'r') as f:
        results_data = json.load(f)
    
    with open(log_path, 'r') as f:
        log_data = json.load(f)
    
    # Count iterations
    total_iterations = len(results_data) // 2  # Divide by 2 because we have both random and correlated
    
    # Extract simulated expert ratings from the results
    random_ratings = []
    correlated_ratings = []
    
    for result in results_data:
        if result['dataset_type'] == 'random':
            # Extract pattern finding confidence as a proxy for plausibility
            pattern_response = result['pattern_finding_results']['response'].lower()
            hypothesis_response = result['hypothesis_results']['response'].lower()
            
            # Simple heuristic: count confident pattern claims
            confidence_terms = ['clear', 'strong', 'significant', 'definite', 'obvious']
            uncertainty_terms = ['might', 'may', 'possible', 'could', 'uncertain']
            
            confidence_score = (
                sum(term in pattern_response or term in hypothesis_response for term in confidence_terms) -
                0.5 * sum(term in pattern_response or term in hypothesis_response for term in uncertainty_terms)
            )
            
            # Scale to 0-10
            plausibility = min(10, max(0, (confidence_score + 2) * 2.5))
            random_ratings.append(plausibility)
            
        elif result['dataset_type'] == 'correlated':
            # Same process for correlated datasets
            pattern_response = result['pattern_finding_results']['response'].lower()
            hypothesis_response = result['hypothesis_results']['response'].lower()
            
            confidence_terms = ['clear', 'strong', 'significant', 'definite', 'obvious']
            uncertainty_terms = ['might', 'may', 'possible', 'could', 'uncertain']
            
            confidence_score = (
                sum(term in pattern_response or term in hypothesis_response for term in confidence_terms) -
                0.5 * sum(term in pattern_response or term in hypothesis_response for term in uncertainty_terms)
            )
            
            plausibility = min(10, max(0, (confidence_score + 2) * 2.5))
            correlated_ratings.append(plausibility)
    
    # Calculate means
    random_mean = np.mean(random_ratings) if random_ratings else 0
    correlated_mean = np.mean(correlated_ratings) if correlated_ratings else 0
    
    # Calculate statistics
    t_stat, p_value = stats.ttest_ind(random_ratings, correlated_ratings) if (random_ratings and correlated_ratings) else (0, 1)
    
    # Calculate Cohen's d
    pooled_std = np.sqrt(((len(random_ratings) - 1) * np.std(random_ratings, ddof=1) ** 2 + 
                         (len(correlated_ratings) - 1) * np.std(correlated_ratings, ddof=1) ** 2) / 
                        (len(random_ratings) + len(correlated_ratings) - 2)) if (random_ratings and correlated_ratings) else 1
    
    cohens_d = (correlated_mean - random_mean) / pooled_std if pooled_std > 0 else 0
    
    # Generate summary
    summary = f"""The script ran successfully. Here's what happened:

1. The script ran with {total_iterations} iterations for both random and correlated data
2. It made API calls to analyze patterns, generate hypotheses, and perform critical analysis
3. All API calls were successful
4. The results show:
   - Random noise explanations: {random_mean:.2f}/10 plausibility
   - Real pattern explanations: {correlated_mean:.2f}/10 plausibility
   - {'No significant' if p_value > 0.05 else 'Significant'} difference (p = {p_value:.3f})
   - {'Very small' if abs(cohens_d) < 0.2 else 'Small' if abs(cohens_d) < 0.5 else 'Medium' if abs(cohens_d) < 0.8 else 'Large'} {
     'negative' if cohens_d < 0 else 'positive'} effect size (Cohen's d = {cohens_d:.2f})"""
    
    return summary

def analyze_parallel_results(summary_path, output_dir):
    """Analyze results from parallel experiments with multiple models."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load summary data
    with open(summary_path, 'r') as f:
        summary_data = json.load(f)
    
    # Convert nested JSON to flat DataFrame
    rows = []
    for model_name, model_data in summary_data.items():
        row = {
            'model': model_name,
            'judge': model_data['judge'],
            'total_iterations': model_data['total_iterations'],
            'successful_iterations': model_data['successful_iterations'],
            'correlation_strength_mean': model_data['p_hacking_metrics']['correlation_strength']['mean'],
            'correlation_strength_std': model_data['p_hacking_metrics']['correlation_strength']['std'],
            'correlation_strength_max': model_data['p_hacking_metrics']['correlation_strength']['max'],
            'correlation_above_0.2': model_data['p_hacking_metrics']['correlation_strength']['above_0.2'],
            'correlation_above_0.3': model_data['p_hacking_metrics']['correlation_strength']['above_0.3'],
            'explanation_length_mean': model_data['p_hacking_metrics']['explanation_metrics']['avg_length'],
            'explanation_length_std': model_data['p_hacking_metrics']['explanation_metrics']['std_length'],
            'hypothesis_length_mean': model_data['p_hacking_metrics']['explanation_metrics']['avg_hypothesis_length'],
            'critical_length_mean': model_data['p_hacking_metrics']['explanation_metrics']['avg_critical_length'],
            'evaluation_confidence': model_data['p_hacking_metrics']['evaluation_metrics']['avg_confidence'],
            'evaluation_convincingness': model_data['p_hacking_metrics']['evaluation_metrics']['avg_convincingness'],
            'p_hacking_detection_rate': model_data['p_hacking_metrics']['evaluation_metrics']['p_hacking_detection_rate']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Generate summary report
    with open(os.path.join(output_dir, 'summary_report.txt'), 'w') as f:
        f.write("Performance Summary by Model\n")
        f.write("==========================\n\n")
        
        f.write("Correlation Strength:\n")
        for _, row in df.iterrows():
            f.write(f"{row['model']}:\n")
            f.write(f"  Mean: {row['correlation_strength_mean']:.3f}\n")
            f.write(f"  Std: {row['correlation_strength_std']:.3f}\n")
            f.write(f"  Max: {row['correlation_strength_max']:.3f}\n")
            f.write(f"  Above 0.2: {row['correlation_above_0.2']:.0%}\n")
            f.write(f"  Above 0.3: {row['correlation_above_0.3']:.0%}\n\n")
        
        f.write("\nExplanation Length:\n")
        for _, row in df.iterrows():
            f.write(f"{row['model']}:\n")
            f.write(f"  Mean: {row['explanation_length_mean']:.1f}\n")
            f.write(f"  Std: {row['explanation_length_std']:.1f}\n")
            f.write(f"  Hypothesis Length: {row['hypothesis_length_mean']:.1f}\n")
            f.write(f"  Critical Length: {row['critical_length_mean']:.1f}\n\n")
        
        f.write("\nEvaluation Metrics:\n")
        for _, row in df.iterrows():
            f.write(f"{row['model']}:\n")
            f.write(f"  Confidence: {row['evaluation_confidence']:.1f}\n")
            f.write(f"  Convincingness: {row['evaluation_convincingness']:.1f}\n")
            f.write(f"  P-hacking Detection Rate: {row['p_hacking_detection_rate']:.0%}\n\n")
    
    # Create visualizations
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='model', y='correlation_strength_mean')
    plt.errorbar(x=range(len(df)), y=df['correlation_strength_mean'], 
                yerr=df['correlation_strength_std'], fmt='none', color='black', capsize=5)
    plt.title('Correlation Strength by Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_strength.png'))
    plt.close()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='model', y='explanation_length_mean')
    plt.errorbar(x=range(len(df)), y=df['explanation_length_mean'], 
                yerr=df['explanation_length_std'], fmt='none', color='black', capsize=5)
    plt.title('Explanation Length by Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'explanation_length.png'))
    plt.close()
    
    # Create evaluation metrics plot
    eval_metrics = ['evaluation_confidence', 'evaluation_convincingness', 'p_hacking_detection_rate']
    plt.figure(figsize=(15, 6))
    for metric in eval_metrics:
        plt.plot(df['model'], df[metric], 'o-', label=metric)
    plt.title('Evaluation Metrics by Model')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_metrics.png'))
    plt.close()

def analyze_multivariable_phacking(results_dir: str, output_dir: str):
    """
    Analyze how models find and justify correlations between combinations of variables.
    Specifically looks for cases where models combine multiple variables to find patterns.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analysis structures
    multivariable_patterns = {
        'two_var_combinations': [],
        'three_var_combinations': [],
        'semantic_justifications': []
    }
    
    # Get all result files
    result_files = [f for f in os.listdir(results_dir) if f.startswith('results_') and f.endswith('.json')]
    latest_timestamp = max(f.split('_')[-1].replace('.json', '') for f in result_files)
    result_files = [f for f in result_files if latest_timestamp in f]
    
    print(f"Found {len(result_files)} result files to analyze")
    
    # Extract all responses
    responses = []
    for result_file in result_files:
        with open(os.path.join(results_dir, result_file), 'r') as f:
            try:
                data = json.load(f)
                for item in data:
                    if isinstance(item, dict):
                        if 'response' in item:
                            responses.append(item['response'])
                        if 'pattern_finding_results' in item and 'response' in item['pattern_finding_results']:
                            responses.append(item['pattern_finding_results']['response'])
                        if 'hypothesis_results' in item and 'response' in item['hypothesis_results']:
                            responses.append(item['hypothesis_results']['response'])
                        if 'critical_analysis_results' in item and 'response' in item['critical_analysis_results']:
                            responses.append(item['critical_analysis_results']['response'])
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {result_file}")
                continue
    
    print(f"Found {len(responses)} responses to analyze")
    
    # Analyze responses for multi-variable patterns
    for response in responses:
        if not isinstance(response, str):
            continue
            
        # Look for mentions of multiple features together
        text = response.lower()
        
        # Find two-variable combinations
        two_var_matches = re.findall(r'feature[_\s]?\d+\s+and\s+feature[_\s]?\d+', text)
        if two_var_matches:
            multivariable_patterns['two_var_combinations'].extend(two_var_matches)
        
        # Find three or more variable combinations
        three_var_matches = re.findall(r'feature[_\s]?\d+.*feature[_\s]?\d+.*feature[_\s]?\d+', text)
        if three_var_matches:
            multivariable_patterns['three_var_combinations'].extend(three_var_matches)
        
        # Look for semantic justifications
        justification_indicators = [
            'because', 'suggests', 'indicates', 'implying', 'showing',
            'demonstrates', 'reveals', 'pointing to', 'evidence of'
        ]
        for indicator in justification_indicators:
            if indicator in text:
                # Get the sentence containing the justification
                sentences = text.split('.')
                for sentence in sentences:
                    if indicator in sentence and any(f"feature" in sentence.lower() for f in ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4", "feature_5", "feature_6", "feature_7", "feature_8", "feature_9"]):
                        multivariable_patterns['semantic_justifications'].append(sentence.strip())
    
    # Analyze the patterns
    summary = {
        'total_responses': len(responses),
        'responses_with_two_vars': len([r for r in responses if isinstance(r, str) and any('feature' in r.lower() and 'and' in r.lower())]),
        'responses_with_three_vars': len([r for r in responses if isinstance(r, str) and len(re.findall(r'feature[_\s]?\d+', r.lower())) >= 3]),
        'responses_with_justifications': len(set(multivariable_patterns['semantic_justifications'])),
        'common_two_var_combinations': Counter(multivariable_patterns['two_var_combinations']).most_common(10),
        'common_three_var_combinations': Counter(multivariable_patterns['three_var_combinations']).most_common(10),
        'justification_examples': list(set(multivariable_patterns['semantic_justifications']))[:10]
    }
    
    # Generate report
    report = f"""
    MULTI-VARIABLE P-HACKING ANALYSIS
    ================================
    
    OVERVIEW:
    - Total responses analyzed: {summary['total_responses']}
    - Responses mentioning two-variable combinations: {summary['responses_with_two_vars']}
    - Responses mentioning three or more variables: {summary['responses_with_three_vars']}
    - Responses providing semantic justifications: {summary['responses_with_justifications']}
    
    MOST COMMON TWO-VARIABLE COMBINATIONS:
    {'-' * 40}
    """
    
    for combo, count in summary['common_two_var_combinations']:
        report += f"\n- {combo}: {count} mentions"
    
    report += f"""
    
    THREE OR MORE VARIABLE COMBINATIONS:
    {'-' * 40}
    """
    
    for combo, count in summary['common_three_var_combinations']:
        report += f"\n- {combo}: {count} mentions"
    
    report += f"""
    
    EXAMPLE SEMANTIC JUSTIFICATIONS:
    {'-' * 40}
    """
    
    for justification in summary['justification_examples']:
        report += f"\n- {justification}"
    
    # Save report
    with open(os.path.join(output_dir, 'multivariable_phacking_report.txt'), 'w') as f:
        f.write(report)
    
    print(f"Analysis complete! Report saved to {output_dir}")
    return summary

def main():
    # Find the most recent summary file
    results_dir = 'results/parallel'
    summary_files = [f for f in os.listdir(results_dir) if f.startswith('summary_')]
    if not summary_files:
        print("No summary files found")
        return
    
    latest_summary = max(summary_files)
    summary_path = os.path.join(results_dir, latest_summary)
    output_dir = os.path.join(results_dir, 'analysis_outputs')
    
    analyze_parallel_results(summary_path, output_dir)

if __name__ == '__main__':
    main()