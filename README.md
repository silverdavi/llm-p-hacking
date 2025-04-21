# LLM P-Hacking Experiment

This repository contains the code and results for an experiment testing large language models' propensity to find and explain correlations in synthetic datasets.

## Overview

The experiment investigates how language models analyze datasets with different types of relationships (quadratic, exponential, mediation, etc.) and evaluates their ability to correctly identify relationship types, explain mechanisms, and avoid finding spurious patterns.

## Key Components

- **Relationship Generator**: Creates synthetic datasets with various relationship types
- **Config Manager**: Handles experiment configuration and logging
- **Parallel Experiment Runner**: Executes multiple iterations in parallel batches
- **Analysis Script**: Evaluates and visualizes results

## Relationships Tested

The experiment tests language models on datasets with the following relationship types:

- **Quadratic**: y = ax² + noise
- **Interaction**: y = ax₁x₂ + noise
- **Mediation**: m = ax + noise; y = bm + noise
- **Temporal**: y = a·sin(t) + noise
- **Cyclic**: y = a·cos(t) + noise
- **Exponential**: y = a·exp(x) + noise
- **Threshold**: y = a·(x > threshold) + noise

## Results

The latest experiment results are packaged in a ZIP file that includes:
- Raw results in JSON format
- Visualizations of model performance by relationship type
- Detailed qualitative examples and analysis

## How to Reproduce

1. Install the required dependencies:
```
pip install -r requirements.txt
```

2. Configure experiment parameters in `config/default_experiment.yaml`

3. Run the experiment:
```
python run_parallel_experiment.py
```

4. Analyze the results:
```
python analyze_results.py
```

5. Package results for publication:
```
python package_results.py
```

## Requirements

- Python 3.8+
- numpy, pandas, matplotlib, seaborn
- OpenAI API key (set as environment variable)
- tabulate for formatting tables

## Project Structure

```
├── config/                  # Configuration files
│   └── default_experiment.yaml
├── utils/                   # Utility modules
│   ├── api/                 # API integration
│   ├── experiment/          # Experiment components
│   └── ...                  
├── experiments/             # Experiment results
├── run_parallel_experiment.py  # Main experiment script
├── analyze_results.py       # Analysis script
└── package_results.py       # Package results for publication
```

## Findings

The experiment reveals several patterns in language model behavior:

1. Performance varies significantly by relationship type, with models generally handling simpler relationships (like quadratic) better than complex ones (like temporal or exponential).

2. Models tend to default to linear relationship explanations when uncertain.

3. The rate of spurious pattern identification increases with relationship complexity.

4. Performance metrics show that relationship identification (average score: ~46) is generally better than mechanism explanation (average score: ~40).

## Future Work

Future experiments could explore:
- Testing with different model sizes and architectures
- Adding more complex multi-variable relationships
- Testing for domain-specific knowledge in specialized fields
- Investigating the effect of data scale on model performance
