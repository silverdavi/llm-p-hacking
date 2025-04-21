"""
Base experiment manager for running p-hacking experiments.
"""

import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json
import numpy as np
import re
from scipy import stats
from utils.api.parallel_llm import ParallelLLMApi

logger = logging.getLogger(__name__)

class ExperimentManager:
    """Manages execution of a single experiment iteration"""

    def __init__(
            self,
            num_iterations: int = 1,
            output_dir: str = "results",
            save_raw: bool = True,
            save_summary: bool = True
    ):
        """
        Initialize experiment manager.

        Args:
            num_iterations: Number of iterations to run
            output_dir: Directory to save results
            save_raw: Whether to save raw data
            save_summary: Whether to save summaries
        """
        self.num_iterations = num_iterations
        self.output_dir = Path(output_dir)
        self.save_raw = save_raw
        self.save_summary = save_summary
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        self.raw_results = []
        self.summaries = []
        
        # Initialize API
        self.api = None

    async def run_experiment(self) -> Dict[str, Any]:
        """Run a single experiment iteration"""
        logger.info(f"Starting experiment with {self.num_iterations} iterations")
        
        # Initialize API if not already set
        if not self.api:
            self.api = ParallelLLMApi()
        
        # Run iterations
        for i in range(self.num_iterations):
            logger.info(f"Running iteration {i + 1}/{self.num_iterations}")
            
            # Generate random and correlated datasets
            random_data = self._generate_random_data()
            correlated_data = self._generate_correlated_data()
            
            # Analyze datasets
            random_results = await self._analyze_dataset(random_data, "random")
            correlated_results = await self._analyze_dataset(correlated_data, "correlated")
            
            # Store results
            iteration_results = {
                "iteration": i,
                "random_results": random_results,
                "correlated_results": correlated_results,
                "timestamp": datetime.now().isoformat()
            }
            self.raw_results.append(iteration_results)
            
            # Generate summary
            summary = self._generate_summary(iteration_results)
            self.summaries.append(summary)
            
            # Save intermediate results
            if self.save_raw:
                self._save_raw_results()
            if self.save_summary:
                self._save_summaries()
        
        logger.info("Experiment completed")
        return {
            "raw_results": self.raw_results,
            "summaries": self.summaries
        }

    def _generate_random_data(self) -> pd.DataFrame:
        """Generate random dataset"""
        # Generate random data with 10 features and 100 samples
        data = np.random.normal(0, 1, size=(100, 10))
        return pd.DataFrame(data, columns=[f"feature_{i}" for i in range(10)])

    def _generate_correlated_data(self) -> pd.DataFrame:
        """Generate correlated dataset"""
        # Generate base random data
        data = np.random.normal(0, 1, size=(100, 10))
        
        # Add correlations between some features
        data[:, 1] = 0.7 * data[:, 0] + 0.3 * np.random.normal(0, 1, 100)
        data[:, 3] = 0.6 * data[:, 2] + 0.4 * np.random.normal(0, 1, 100)
        data[:, 5] = 0.5 * data[:, 4] + 0.5 * np.random.normal(0, 1, 100)
        
        return pd.DataFrame(data, columns=[f"feature_{i}" for i in range(10)])

    async def _analyze_dataset(
            self,
            data: pd.DataFrame,
            dataset_type: str
    ) -> Dict[str, Any]:
        """Analyze dataset using LLM"""
        # Convert data to string for LLM
        data_str = data.to_string()
        
        # Create prompt with more explicit instructions
        prompt = f"""
        Analyze the following dataset and identify any patterns or correlations.
        Dataset type: {dataset_type}
        
        Data:
        {data_str}
        
        Please provide your analysis in the following format:
        
        PATTERNS:
        [List any patterns or correlations you observe]
        
        P-VALUE:
        [Provide a single numerical p-value for the most significant correlation, e.g., 0.023]
        
        SIGNIFICANCE:
        [State whether patterns are statistically significant (p < 0.05) as YES or NO]
        """
        
        # Get LLM response
        response = await self.api.process_batch([prompt])
        response_text = response[0] if response else ""
        
        # Parse response
        try:
            # Extract p-value using more robust parsing
            p_value = None
            is_significant = False
            
            # Look for p-value in different formats
            if "P-VALUE:" in response_text:
                p_value_section = response_text.split("P-VALUE:")[1].split("\n")[0].strip()
                # Try to extract first number from the section
                numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", p_value_section)
                if numbers:
                    p_value = float(numbers[0])
            
            # If no p-value found, try alternative formats
            if p_value is None:
                # Look for p-value mentioned in text
                p_value_matches = re.findall(r"p-value.*?(\d+\.\d+)", response_text, re.IGNORECASE)
                if p_value_matches:
                    p_value = float(p_value_matches[0])
            
            # Determine significance
            if "SIGNIFICANCE:" in response_text:
                significance_text = response_text.split("SIGNIFICANCE:")[1].split("\n")[0].lower()
                is_significant = "yes" in significance_text or "significant" in significance_text
            elif p_value is not None:
                is_significant = p_value < 0.05
            
            # If still no p-value, generate one based on correlations
            if p_value is None:
                # Calculate actual correlations in the data
                corr_matrix = data.corr()
                # Get the strongest correlation (excluding self-correlations)
                strongest_corr = abs(corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)]).max()
                # Convert correlation to a p-value using Fisher transformation
                n = len(data)
                z = np.arctanh(strongest_corr)
                p_value = 2 * (1 - stats.norm.cdf(abs(z) * np.sqrt(n-3)))
            
            return {
                "p_value": float(p_value) if p_value is not None else None,
                "is_significant": bool(is_significant),
                "response": response_text
            }
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            # Fallback to correlation-based p-value
            try:
                corr_matrix = data.corr()
                strongest_corr = abs(corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)]).max()
                n = len(data)
                z = np.arctanh(strongest_corr)
                p_value = 2 * (1 - stats.norm.cdf(abs(z) * np.sqrt(n-3)))
                
                return {
                    "p_value": float(p_value),
                    "is_significant": p_value < 0.05,
                    "response": response_text,
                    "note": "Used correlation-based p-value due to parsing error"
                }
            except Exception as e2:
                logger.error(f"Error calculating fallback p-value: {str(e2)}")
                return {
                    "p_value": None,
                    "is_significant": False,
                    "response": response_text,
                    "note": "Failed to parse response and calculate fallback"
                }

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of iteration results"""
        return {
            "iteration": results["iteration"],
            "random_p_value": results["random_results"]["p_value"],
            "random_significant": results["random_results"]["is_significant"],
            "correlated_p_value": results["correlated_results"]["p_value"],
            "correlated_significant": results["correlated_results"]["is_significant"],
            "timestamp": results["timestamp"]
        }

    def _save_raw_results(self):
        """Save raw results to CSV"""
        if not self.raw_results:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df = pd.DataFrame(self.raw_results)
        df.to_csv(self.output_dir / f"raw_results_{timestamp}.csv", index=False)
        logger.info(f"Saved raw results to {self.output_dir}/raw_results_{timestamp}.csv")

    def _save_summaries(self):
        """Save summaries to JSON"""
        if not self.summaries:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(self.output_dir / f"summaries_{timestamp}.json", "w") as f:
            json.dump(self.summaries, f, indent=2)
        logger.info(f"Saved summaries to {self.output_dir}/summaries_{timestamp}.json") 