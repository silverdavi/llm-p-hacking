# LLM P-Hacking Experiment

This project investigates LLMs' propensity for semantic p-hacking by analyzing their responses to synthetic datasets. It tests whether LLMs tend to find and report patterns in both random noise and genuinely correlated data.

## Features

- Generate synthetic datasets with controlled correlations
- Test LLM responses across different prompt types:
  - Pattern finding
  - Hypothesis generation
  - Critical analysis
- Comprehensive analysis tools including:
  - Statistical analysis of LLM responses
  - Visualization of results
  - Pattern claim rate analysis
  - Effect size calculations

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=your_api_key_here
```

## Running Experiments

1. Run the main experiment:
```bash
python run_tests_mini.py
```
This will:
- Generate synthetic datasets (random and correlated)
- Query the LLM with different prompt types
- Save results in `semantic_phacking_experiments_mini/[timestamp]`

2. Analyze the results:
```bash
python utils/analyze_results.py
```
This generates:
- Statistical analysis of LLM responses
- Visualizations of patterns and trends
- Detailed and concise summaries
- Saved outputs in `analysis_outputs` directory

## Project Structure

```
├── run_tests_mini.py          # Main experiment script
├── requirements.txt           # Project dependencies
├── utils/
│   ├── api/                  # API interaction utilities
│   │   ├── util_call.py     # OpenAI API wrapper
│   │   └── llm_api.py       # LLM API implementation
│   ├── analyze_results.py    # Results analysis tools
│   └── config/              # Configuration files
├── .env.example             # Example environment variables
└── README.md               # This file
```

## Results Format

The experiment generates several types of outputs:
1. `experiment_results.json`: Raw experiment results
2. `interaction_log.json`: Detailed API interaction logs
3. Analysis outputs including:
   - Statistical summaries
   - Visualization plots
   - Pattern analysis reports

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
