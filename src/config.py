"""Configuration management for PromptXpert."""

import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for language models."""
    primary_model: str = "llama3-8b-8192"  # Groq model for main tasks
    judge_model: str = "llama3-70b-8192"   # Groq model for judging (different size)
    primary_temperature: float = 0.7
    judge_temperature: float = 0.3
    max_tokens: int = 1024
    api_key_env: str = "GROQ_API_KEY"


@dataclass
class OptimizationConfig:
    """Configuration for DSPy optimization."""
    max_labeled_demos: int = 4
    max_bootstrapped_demos: int = 4
    auto_mode: str = "medium"  # light, medium, heavy
    num_trials: int = 20
    minibatch_size: int = 4


@dataclass
class DataConfig:
    """Configuration for dataset handling."""
    dataset_path: str = "data/prompts_dataset.csv"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42


@dataclass
class MLflowConfig:
    """Configuration for MLflow experiment tracking."""
    experiment_name: str = "PromptXpert_Optimization"
    tracking_uri: Optional[str] = None  # Uses local if None
    artifact_location: Optional[str] = None


@dataclass
class MetricsConfig:
    """Configuration for evaluation metrics."""
    clarity_weight: float = 0.35
    specificity_weight: float = 0.35
    brevity_weight: float = 0.15
    safety_weight: float = 0.15


@dataclass
class PromptXpertConfig:
    """Main configuration class."""
    model: ModelConfig = None
    optimization: OptimizationConfig = None
    data: DataConfig = None
    mlflow: MLflowConfig = None
    metrics: MetricsConfig = None
    
    # Global settings
    random_seed: int = 42
    output_dir: str = "outputs"
    cache_enabled: bool = True
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Validate configuration and create output directory."""
        # Initialize defaults if None
        if self.model is None:
            self.model = ModelConfig()
        if self.optimization is None:
            self.optimization = OptimizationConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.mlflow is None:
            self.mlflow = MLflowConfig()
        if self.metrics is None:
            self.metrics = MetricsConfig()
        
        # Ensure ratios sum to 1.0
        total_ratio = self.data.train_ratio + self.data.val_ratio + self.data.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Data split ratios must sum to 1.0, got {total_ratio}")
        
        # Create output directory
        Path(self.output_dir).mkdir(exist_ok=True)
        
        # Validate weights sum to 1.0
        total_weight = (self.metrics.clarity_weight + self.metrics.specificity_weight + 
                       self.metrics.brevity_weight + self.metrics.safety_weight)
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Metric weights must sum to 1.0, got {total_weight}")


def load_config_from_env() -> PromptXpertConfig:
    """Load configuration with environment variable overrides."""
    config = PromptXpertConfig()
    
    # Override with environment variables if present
    if os.getenv("PROMPTXPERT_PRIMARY_MODEL"):
        config.model.primary_model = os.getenv("PROMPTXPERT_PRIMARY_MODEL")
    
    if os.getenv("PROMPTXPERT_JUDGE_MODEL"):
        config.model.judge_model = os.getenv("PROMPTXPERT_JUDGE_MODEL")
    
    if os.getenv("PROMPTXPERT_EXPERIMENT_NAME"):
        config.mlflow.experiment_name = os.getenv("PROMPTXPERT_EXPERIMENT_NAME")
    
    if os.getenv("MLFLOW_TRACKING_URI"):
        config.mlflow.tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    
    return config