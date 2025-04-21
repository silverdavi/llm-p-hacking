"""
Configuration management for p-hacking experiments.
"""

import json
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import yaml
from dataclasses import dataclass, asdict

@dataclass
class ExperimentConfig:
    """Configuration for a p-hacking experiment"""
    # Model settings
    analysis_model: str
    judge_model: str
    
    # Experiment parameters
    num_iterations: int
    batch_size: int
    n_samples: int
    n_features: int
    
    # Relationship parameters
    relationship_types: list[str]
    relationship_strengths: Dict[str, float]
    noise_levels: Dict[str, float]
    
    # API settings
    api_retry_attempts: int = 3
    api_retry_delay: float = 1.0
    api_timeout: float = 30.0
    
    # Logging settings
    log_level: str = "INFO"
    log_rotation_size: int = 1024 * 1024  # 1MB
    log_backup_count: int = 5

class ConfigManager:
    """Manages experiment configuration and logging"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config: Optional[ExperimentConfig] = None
        
        # Set up base directories
        self.base_dir = Path("experiments")
        self.experiment_dir = self.base_dir / self.experiment_id
        self.log_dir = self.experiment_dir / "logs"
        self.results_dir = self.experiment_dir / "results"
        self.config_dir = self.experiment_dir / "config"
        
        # Create directories
        self._setup_directories()
        
        if config_path:
            self.load_config(config_path)
        
    def _setup_directories(self):
        """Create necessary directories"""
        for directory in [self.log_dir, self.results_dir, self.config_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
    def setup_logging(self):
        """Configure logging with rotation and formatting"""
        log_file = self.log_dir / f"experiment_{self.experiment_id}.log"
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.config.log_level if self.config else "INFO")
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.config.log_rotation_size if self.config else 1024*1024,
            backupCount=self.config.log_backup_count if self.config else 5
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
    def load_config(self, config_path: str):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            self.config = ExperimentConfig(**config_dict)
            
        # Save config to experiment directory
        self.save_config()
        
    def save_config(self):
        """Save current configuration"""
        if not self.config:
            return
            
        config_file = self.config_dir / f"config_{self.experiment_id}.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(asdict(self.config), f, default_flow_style=False)
            
    def get_default_config(self) -> ExperimentConfig:
        """Get default configuration"""
        return ExperimentConfig(
            analysis_model="gpt-4",
            judge_model="gpt-4",
            num_iterations=10,
            batch_size=5,
            n_samples=100,
            n_features=10,
            relationship_types=[
                "quadratic",
                "interaction",
                "mediation",
                "temporal",
                "threshold",
                "cyclic",
                "exponential"
            ],
            relationship_strengths={
                "quadratic": 0.7,
                "interaction": 0.6,
                "mediation": 0.5,
                "temporal": 0.8,
                "threshold": 0.6,
                "cyclic": 0.7,
                "exponential": 0.6
            },
            noise_levels={
                "quadratic": 0.3,
                "interaction": 0.4,
                "mediation": 0.3,
                "temporal": 0.2,
                "threshold": 0.3,
                "cyclic": 0.3,
                "exponential": 0.4
            }
        )
        
    def create_experiment_metadata(self) -> Dict[str, Any]:
        """Create metadata for the experiment"""
        return {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now().isoformat(),
            "config": asdict(self.config) if self.config else None,
            "directories": {
                "logs": str(self.log_dir),
                "results": str(self.results_dir),
                "config": str(self.config_dir)
            }
        }
        
    def save_experiment_metadata(self, metadata: Dict[str, Any]):
        """Save experiment metadata"""
        metadata_file = self.experiment_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def get_experiment_paths(self) -> Dict[str, Path]:
        """Get important experiment paths"""
        return {
            "base": self.base_dir,
            "experiment": self.experiment_dir,
            "logs": self.log_dir,
            "results": self.results_dir,
            "config": self.config_dir
        } 