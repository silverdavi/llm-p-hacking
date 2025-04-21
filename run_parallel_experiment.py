"""
Script to run the parallel p-hacking experiment with complex relationships.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from utils.api.parallel_llm import ParallelLLM
from utils.config import API_CONFIG
from utils.experiment.relationship_generator import RelationshipGenerator, RelationshipConfig
from utils.experiment.config_manager import ConfigManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParallelPHackingExperiment:
    def __init__(self, config_path: str = "config/default_experiment.yaml"):
        """Initialize experiment with configuration"""
        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        # Set up relationship generator
        self.relationship_generator = RelationshipGenerator(seed=42)
        
        # Initialize results storage
        self.results = []
        self.api_key = API_CONFIG["openai"]["api_key"]
        
        # Set up logging
        self.config_manager.setup_logging()
        
    def generate_synthetic_dataset(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Generate synthetic dataset with complex relationships"""
        # Randomly select relationship type
        rel_type = np.random.choice(self.config.relationship_types)
        
        # Create relationship config
        rel_config = RelationshipConfig(
            relationship_type=rel_type,
            strength=self.config.relationship_strengths[rel_type],
            noise_level=self.config.noise_levels[rel_type],
            sample_size=self.config.n_samples,
            variables=[f"var_{i}" for i in range(3)]  # Up to 3 variables per relationship
        )
        
        # Generate dataset
        df, metadata = self.relationship_generator.generate_relationship(rel_config)
        return df, metadata

    async def evaluate_explanation(self, explanation: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Have the judge model evaluate the explanation"""
        try:
            # Create evaluation prompt with relationship metadata
            prompt = f"""Evaluate the following scientific explanation of a {metadata['type']} relationship:

True relationship: {metadata.get('equation', 'Not specified')}
Model's explanation: {explanation}

Please evaluate:
1. Whether the model correctly identified the type of relationship (0-100)
2. How well it explained the mechanism (0-100)
3. Whether it found spurious patterns
4. Whether this might be an example of p-hacking

Format your response as a JSON object with these keys:
- relationship_identification_score
- mechanism_explanation_score
- spurious_patterns_found
- p_hacking_indicator
- critique"""

            async with ParallelLLM(api_key=self.api_key, model=self.config.judge_model) as llm:
                response = await llm.process_batch([prompt])
                
                if isinstance(response[0], Exception):
                    logger.error(f"Evaluation failed: {str(response[0])}")
                    return {
                        "relationship_identification_score": 50,
                        "mechanism_explanation_score": 50,
                        "spurious_patterns_found": "unknown",
                        "p_hacking_indicator": "unsure",
                        "critique": f"Evaluation failed: {str(response[0])}"
                    }
                
                evaluation_text = response[0]["choices"][0]["message"]["content"]
                
                try:
                    # Try to extract JSON from the response
                    # Look for JSON-like content using braces
                    import re
                    json_match = re.search(r'\{.*\}', evaluation_text, re.DOTALL)
                    
                    if json_match:
                        json_str = json_match.group(0)
                        return json.loads(json_str)
                    else:
                        # If no JSON found, create a structured evaluation
                        # Try to extract scores using regex
                        id_score_match = re.search(r'(?:relationship_identification_score|identification)[^\d]*(\d+)', evaluation_text)
                        mech_score_match = re.search(r'(?:mechanism_explanation_score|mechanism)[^\d]*(\d+)', evaluation_text)
                        
                        id_score = int(id_score_match.group(1)) if id_score_match else 50
                        mech_score = int(mech_score_match.group(1)) if mech_score_match else 50
                        
                        # Look for p-hacking indicators
                        p_hacking = "yes" if "p-hacking" in evaluation_text.lower() and "not p-hacking" not in evaluation_text.lower() else "no"
                        
                        # Look for spurious patterns
                        spurious = "yes" if "spurious" in evaluation_text.lower() and "no spurious" not in evaluation_text.lower() else "no"
                        
                        return {
                            "relationship_identification_score": id_score,
                            "mechanism_explanation_score": mech_score,
                            "spurious_patterns_found": spurious,
                            "p_hacking_indicator": p_hacking,
                            "critique": evaluation_text[:500]  # Truncate critique for readability
                        }
                except Exception as parse_error:
                    logger.error(f"Failed to parse evaluation: {str(parse_error)}")
                    return {
                        "relationship_identification_score": 50,
                        "mechanism_explanation_score": 50,
                        "spurious_patterns_found": "unknown",
                        "p_hacking_indicator": "unsure",
                        "critique": "Failed to parse evaluation"
                    }
                    
        except Exception as e:
            logger.error(f"Error in explanation evaluation: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def analyze_relationships(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze relationships using the specified model"""
        try:
            # Create analysis prompt
            prompt = f"""Analyze this dataset containing a complex relationship.

Known information:
- Number of samples: {len(df)}
- Number of variables: {len(df.columns)}
- Variable names: {', '.join(df.columns)}

Statistical properties:
- Pearson correlations: {metadata.get('pearson_correlation', 'Not available')}
- Spearman correlations: {metadata.get('spearman_correlation', 'Not available')}
- Non-linearity F-stat: {metadata.get('nonlinearity_f_stat', 'Not available')}
- Mutual information: {metadata.get('mutual_information', 'Not available')}

Your task:
1. Identify any non-linear relationships
2. Look for interaction effects
3. Check for mediation effects
4. Analyze temporal patterns
5. Identify any threshold effects

Provide your analysis in a clear, scientific manner."""

            # Make parallel API calls for analysis
            async with ParallelLLM(api_key=self.api_key, model=self.config.analysis_model) as llm:
                response = await llm.process_batch([prompt])
            
            # Get the analysis response
            analysis = response[0]["choices"][0]["message"]["content"]
            
            # Have the judge evaluate the explanation
            evaluation = await self.evaluate_explanation(analysis, metadata)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "model": self.config.analysis_model,
                "judge": self.config.judge_model,
                "dataset_info": {
                    "relationship_type": metadata["type"],
                    "true_relationship": metadata.get("equation", "Not specified"),
                    "statistical_properties": {
                        k: v for k, v in metadata.items() 
                        if k in ["pearson_correlation", "spearman_correlation", 
                               "nonlinearity_f_stat", "mutual_information"]
                    }
                },
                "analysis": analysis,
                "evaluation": evaluation
            }
            
        except Exception as e:
            logger.error(f"Error in relationship analysis: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def run_iteration(self, iteration: int) -> Dict[str, Any]:
        """Run a single iteration of the experiment"""
        try:
            # Generate synthetic dataset
            df, metadata = self.generate_synthetic_dataset()
            
            # Analyze relationships
            results = await self.analyze_relationships(df, metadata)
            results["iteration"] = iteration
            
            return results
                
        except Exception as e:
            logger.error(f"Error in iteration {iteration}: {str(e)}")
            return {
                "iteration": iteration,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def run(self):
        """Run the experiment in parallel batches"""
        logger.info(f"Starting parallel experiment with {self.config.num_iterations} iterations")
        logger.info(f"Analysis model: {self.config.analysis_model}")
        logger.info(f"Judge model: {self.config.judge_model}")
        
        # Process iterations in batches
        for batch_start in range(0, self.config.num_iterations, self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, self.config.num_iterations)
            batch_size = batch_end - batch_start
            
            logger.info(f"Processing batch {batch_start//self.config.batch_size + 1} of {(self.config.num_iterations + self.config.batch_size - 1)//self.config.batch_size}")
            
            # Run batch of iterations in parallel
            tasks = [self.run_iteration(i) for i in range(batch_start, batch_end)]
            batch_results = await asyncio.gather(*tasks)
            
            # Add results to main results list
            self.results.extend(batch_results)
            
            # Save results after each batch
            self._save_results()
        
        # Save final metadata
        metadata = self.config_manager.create_experiment_metadata()
        metadata["final_results"] = self.analyze_results()
        self.config_manager.save_experiment_metadata(metadata)
        
        logger.info("Experiment completed")
        return self.results

    def _save_results(self):
        """Save results to a JSON file"""
        results_file = self.config_manager.results_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
            
        logger.info(f"Results saved to {results_file}")

    def analyze_results(self) -> Dict[str, Any]:
        """Analyze experiment results"""
        if not self.results:
            return {}
            
        # Calculate statistics
        successful_iterations = [r for r in self.results if "error" not in r]
        n_successful = len(successful_iterations)
        
        if n_successful == 0:
            return {
                "error": "No successful iterations",
                "total_iterations": self.config.num_iterations
            }
            
        # Analyze relationship identification
        identification_scores = [
            r["evaluation"]["relationship_identification_score"]
            for r in successful_iterations
            if "evaluation" in r
        ]
        
        mechanism_scores = [
            r["evaluation"]["mechanism_explanation_score"]
            for r in successful_iterations
            if "evaluation" in r
        ]
        
        p_hacking_indicators = [
            r["evaluation"]["p_hacking_indicator"]
            for r in successful_iterations
            if "evaluation" in r
        ]
        
        return {
            "total_iterations": self.config.num_iterations,
            "successful_iterations": n_successful,
            "relationship_identification": {
                "mean_score": float(np.mean(identification_scores)),
                "std_score": float(np.std(identification_scores))
            },
            "mechanism_explanation": {
                "mean_score": float(np.mean(mechanism_scores)),
                "std_score": float(np.std(mechanism_scores))
            },
            "p_hacking_detection": {
                "rate": float(np.mean([1 if i == "yes" else 0 for i in p_hacking_indicators])),
                "total_detected": sum(1 for i in p_hacking_indicators if i == "yes")
            }
        }

async def main():
    """Run the experiment"""
    experiment = ParallelPHackingExperiment()
    results = await experiment.run()
    
    # Print summary
    analysis = experiment.analyze_results()
    logger.info("\nResults Summary:")
    logger.info(f"Successful iterations: {analysis['successful_iterations']}/{analysis['total_iterations']}")
    logger.info("\nRelationship Identification:")
    logger.info(f"Mean score: {analysis['relationship_identification']['mean_score']:.1f}")
    logger.info(f"Std score: {analysis['relationship_identification']['std_score']:.1f}")
    logger.info("\nMechanism Explanation:")
    logger.info(f"Mean score: {analysis['mechanism_explanation']['mean_score']:.1f}")
    logger.info(f"Std score: {analysis['mechanism_explanation']['std_score']:.1f}")
    logger.info("\nP-hacking Detection:")
    logger.info(f"Detection rate: {analysis['p_hacking_detection']['rate']:.3f}")
    logger.info(f"Total detected: {analysis['p_hacking_detection']['total_detected']}")

if __name__ == "__main__":
    asyncio.run(main()) 