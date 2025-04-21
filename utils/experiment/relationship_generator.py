"""
Module for generating synthetic datasets with complex relationships for p-hacking experiments.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class RelationshipConfig:
    """Configuration for a synthetic relationship"""
    relationship_type: str
    strength: float
    noise_level: float
    sample_size: int
    variables: List[str]

class RelationshipGenerator:
    """Generates complex relationships for synthetic datasets"""
    
    def __init__(self, seed: int = 42):
        """Initialize the generator with a random seed"""
        self.seed = seed
        np.random.seed(seed)
        self.relationship_registry = {
            "quadratic": self._generate_quadratic,
            "interaction": self._generate_interaction,
            "mediation": self._generate_mediation,
            "temporal": self._generate_temporal,
            "threshold": self._generate_threshold,
            "cyclic": self._generate_cyclic,
            "exponential": self._generate_exponential
        }
        
    def generate_relationship(self, config: RelationshipConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Generate a specific type of relationship"""
        if config.relationship_type not in self.relationship_registry:
            raise ValueError(f"Unknown relationship type: {config.relationship_type}")
            
        generator_func = self.relationship_registry[config.relationship_type]
        df, metadata = generator_func(config)
        
        # Add statistical analysis
        stats_info = self._analyze_relationship(df, config)
        metadata.update(stats_info)
        
        return df, metadata
        
    def _generate_quadratic(self, config: RelationshipConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Generate quadratic relationship: y = ax² + noise"""
        x = np.random.normal(0, 1, config.sample_size)
        y = config.strength * x**2 + config.noise_level * np.random.normal(0, 1, config.sample_size)
        
        df = pd.DataFrame({
            config.variables[0]: x,
            config.variables[1]: y
        })
        
        metadata = {
            "type": "quadratic",
            "true_coefficient": config.strength,
            "noise_level": config.noise_level,
            "equation": f"y = {config.strength:.2f}x² + noise({config.noise_level:.2f})"
        }
        
        return df, metadata
        
    def _generate_interaction(self, config: RelationshipConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Generate interaction effect: y = ax₁x₂ + noise"""
        x1 = np.random.normal(0, 1, config.sample_size)
        x2 = np.random.normal(0, 1, config.sample_size)
        y = config.strength * x1 * x2 + config.noise_level * np.random.normal(0, 1, config.sample_size)
        
        df = pd.DataFrame({
            config.variables[0]: x1,
            config.variables[1]: x2,
            config.variables[2]: y
        })
        
        metadata = {
            "type": "interaction",
            "interaction_strength": config.strength,
            "noise_level": config.noise_level,
            "equation": f"y = {config.strength:.2f}x₁x₂ + noise({config.noise_level:.2f})"
        }
        
        return df, metadata
        
    def _generate_mediation(self, config: RelationshipConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Generate mediation effect: x → m → y"""
        x = np.random.normal(0, 1, config.sample_size)
        m = 0.7 * x + 0.3 * np.random.normal(0, 1, config.sample_size)
        y = config.strength * m + config.noise_level * np.random.normal(0, 1, config.sample_size)
        
        df = pd.DataFrame({
            config.variables[0]: x,
            config.variables[1]: m,
            config.variables[2]: y
        })
        
        metadata = {
            "type": "mediation",
            "direct_effect": 0.7,
            "indirect_effect": config.strength,
            "noise_level": config.noise_level,
            "equation": "m = 0.7x + noise(0.3); y = {strength:.2f}m + noise({noise:.2f})"
        }
        
        return df, metadata
        
    def _generate_temporal(self, config: RelationshipConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Generate temporal pattern with non-linear trend"""
        t = np.linspace(0, 4*np.pi, config.sample_size)
        y = config.strength * np.sin(t) + config.noise_level * np.random.normal(0, 1, config.sample_size)
        
        df = pd.DataFrame({
            config.variables[0]: t,
            config.variables[1]: y
        })
        
        metadata = {
            "type": "temporal",
            "frequency": "4π",
            "amplitude": config.strength,
            "noise_level": config.noise_level,
            "equation": f"y = {config.strength:.2f}sin(t) + noise({config.noise_level:.2f})"
        }
        
        return df, metadata
        
    def _generate_threshold(self, config: RelationshipConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Generate threshold effect with sudden transition"""
        x = np.random.normal(0, 1, config.sample_size)
        threshold = 0.5
        y = np.where(x > threshold, 
                    config.strength * (x - threshold), 
                    0.1 * x) + config.noise_level * np.random.normal(0, 1, config.sample_size)
        
        df = pd.DataFrame({
            config.variables[0]: x,
            config.variables[1]: y
        })
        
        metadata = {
            "type": "threshold",
            "threshold_value": threshold,
            "above_threshold_effect": config.strength,
            "below_threshold_effect": 0.1,
            "noise_level": config.noise_level
        }
        
        return df, metadata
        
    def _generate_cyclic(self, config: RelationshipConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Generate cyclic relationship with multiple frequencies"""
        t = np.linspace(0, 8*np.pi, config.sample_size)
        y = config.strength * (np.sin(t) + 0.5*np.sin(2*t)) + config.noise_level * np.random.normal(0, 1, config.sample_size)
        
        df = pd.DataFrame({
            config.variables[0]: t,
            config.variables[1]: y
        })
        
        metadata = {
            "type": "cyclic",
            "primary_frequency": "8π",
            "secondary_frequency": "4π",
            "amplitude": config.strength,
            "noise_level": config.noise_level
        }
        
        return df, metadata
        
    def _generate_exponential(self, config: RelationshipConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Generate exponential relationship"""
        x = np.random.normal(0, 1, config.sample_size)
        y = config.strength * np.exp(x) + config.noise_level * np.random.normal(0, 1, config.sample_size)
        
        df = pd.DataFrame({
            config.variables[0]: x,
            config.variables[1]: y
        })
        
        metadata = {
            "type": "exponential",
            "coefficient": config.strength,
            "noise_level": config.noise_level,
            "equation": f"y = {config.strength:.2f}exp(x) + noise({config.noise_level:.2f})"
        }
        
        return df, metadata
        
    def _analyze_relationship(self, df: pd.DataFrame, config: RelationshipConfig) -> Dict[str, Any]:
        """Perform statistical analysis on the relationship"""
        stats_info = {}
        
        # Pearson correlation
        corr_matrix = df.corr(method='pearson')
        stats_info['pearson_correlation'] = corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)].tolist()
        
        # Spearman correlation
        spearman_matrix = df.corr(method='spearman')
        stats_info['spearman_correlation'] = spearman_matrix.values[np.triu_indices_from(spearman_matrix, k=1)].tolist()
        
        # Kendall correlation
        kendall_matrix = df.corr(method='kendall')
        stats_info['kendall_correlation'] = kendall_matrix.values[np.triu_indices_from(kendall_matrix, k=1)].tolist()
        
        # Non-linearity test (if applicable)
        if len(df.columns) == 2:
            x = df.iloc[:, 0]
            y = df.iloc[:, 1]
            
            # Test for non-linearity using polynomial fit comparison
            linear_fit = np.polyfit(x, y, 1)
            quad_fit = np.polyfit(x, y, 2)
            
            linear_residuals = y - np.polyval(linear_fit, x)
            quad_residuals = y - np.polyval(quad_fit, x)
            
            f_stat = (np.sum(linear_residuals**2) - np.sum(quad_residuals**2)) / np.sum(quad_residuals**2)
            stats_info['nonlinearity_f_stat'] = float(f_stat)
            
            # Mutual information score
            from sklearn.metrics import mutual_info_score
            stats_info['mutual_information'] = float(mutual_info_score(
                pd.qcut(x, 10, labels=False, duplicates='drop'),
                pd.qcut(y, 10, labels=False, duplicates='drop')
            ))
        
        return stats_info 