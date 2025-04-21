"""
Parallel experiment manager for running multiple iterations of the p-hacking experiment concurrently.
"""

import asyncio
import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json
import numpy as np
from utils.api.parallel_llm import ParallelLLMApi
from utils.experiment.experiment_manager import ExperimentManager

logger = logging.getLogger(__name__)

class ParallelExperimentManager:
    """Manages parallel execution of multiple experiment iterations"""

    def __init__(
            self,
            num_iterations: int = 10,
            batch_size: int = 5,
            max_concurrent: int = 5,
            output_dir: str = "results",
            save_raw: bool = True,
            save_summary: bool = True
    ):
        """
        Initialize parallel experiment manager.

        Args:
            num_iterations: Total number of iterations to run
            batch_size: Number of iterations to process in each batch
            max_concurrent: Maximum number of concurrent API calls
            output_dir: Directory to save results
            save_raw: Whether to save raw data
            save_summary: Whether to save summaries
        """
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.output_dir = Path(output_dir)
        self.save_raw = save_raw
        self.save_summary = save_summary
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        self.raw_results = []
        self.summaries = []
        
        # Initialize API
        self.api = ParallelLLMApi(max_concurrent=max_concurrent)

    async def run_experiment(self):
        """Run the experiment in parallel batches"""
        logger.info(f"Starting parallel experiment with {self.num_iterations} iterations")
        
        # Calculate number of batches
        num_batches = (self.num_iterations + self.batch_size - 1) // self.batch_size
        
        try:
            for batch_num in range(num_batches):
                start_idx = batch_num * self.batch_size
                end_idx = min(start_idx + self.batch_size, self.num_iterations)
                batch_size = end_idx - start_idx
                
                logger.info(f"Processing batch {batch_num + 1}/{num_batches} ({batch_size} iterations)")
                
                # Create batch of experiment managers
                batch_managers = [
                    ExperimentManager(
                        num_iterations=1,
                        output_dir=str(self.output_dir),
                        save_raw=False,
                        save_summary=False
                    )
                    for _ in range(batch_size)
                ]
                
                # Run batch in parallel
                batch_results = await asyncio.gather(*[
                    self._run_single_experiment(manager)
                    for manager in batch_managers
                ])
                
                # Process results
                for result in batch_results:
                    if result:
                        self.raw_results.extend(result.get("raw_results", []))
                        self.summaries.extend(result.get("summaries", []))
                
                # Save intermediate results
                if self.save_raw:
                    self._save_raw_results()
                if self.save_summary:
                    self._save_summaries()
                
                logger.info(f"Completed batch {batch_num + 1}/{num_batches}")
        
            logger.info("Experiment completed")
            return {
                "raw_results": self.raw_results,
                "summaries": self.summaries
            }
            
        except Exception as e:
            logger.error(f"Error running experiment: {str(e)}")
            raise

    async def _run_single_experiment(
            self,
            manager: ExperimentManager
    ) -> Optional[Dict[str, Any]]:
        """Run a single experiment iteration"""
        try:
            # Set the API for this manager
            manager.api = self.api
            return await manager.run_experiment()
        except Exception as e:
            logger.error(f"Error in experiment iteration: {str(e)}")
            return None

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

    def get_statistics(self) -> Dict[str, Any]:
        """Calculate and return experiment statistics"""
        if not self.summaries:
            return {
                "total_iterations": 0,
                "success_rates": {"random": 0.0, "correlated": 0.0},
                "average_p_values": {"random": 0.0, "correlated": 0.0},
                "total_api_calls": 0
            }
            
        try:
            # Convert summaries to DataFrame
            df = pd.DataFrame(self.summaries)
            
            # Calculate success rates, handling missing or invalid data
            success_rates = {
                "random": df["random_significant"].mean() if "random_significant" in df.columns else 0.0,
                "correlated": df["correlated_significant"].mean() if "correlated_significant" in df.columns else 0.0
            }
            
            # Calculate average p-values, handling missing or invalid data
            p_values = {
                "random": df["random_p_value"].mean() if "random_p_value" in df.columns else 0.0,
                "correlated": df["correlated_p_value"].mean() if "correlated_p_value" in df.columns else 0.0
            }
            
            return {
                "total_iterations": len(self.summaries),
                "success_rates": success_rates,
                "average_p_values": p_values,
                "total_api_calls": len(self.raw_results) * 2  # 2 API calls per iteration (random + correlated)
            }
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            return {
                "total_iterations": len(self.summaries),
                "success_rates": {"random": 0.0, "correlated": 0.0},
                "average_p_values": {"random": 0.0, "correlated": 0.0},
                "total_api_calls": len(self.raw_results) * 2
            } 