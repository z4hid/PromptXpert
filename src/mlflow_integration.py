"""MLflow integration for experiment tracking in PromptXpert."""

import mlflow
import mlflow.dspy
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import dspy

from .config import PromptXpertConfig, MLflowConfig

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Class for managing MLflow experiment tracking."""
    
    def __init__(self, config: MLflowConfig):
        """Initialize MLflow tracker.
        
        Args:
            config: MLflow configuration
        """
        self.config = config
        self.experiment_id = None
        self.run_id = None
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Setup MLflow tracking configuration."""
        # Set tracking URI if specified
        if self.config.tracking_uri:
            mlflow.set_tracking_uri(self.config.tracking_uri)
        
        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
            if experiment is None:
                self.experiment_id = mlflow.create_experiment(
                    name=self.config.experiment_name,
                    artifact_location=self.config.artifact_location
                )
            else:
                self.experiment_id = experiment.experiment_id
            
            mlflow.set_experiment(self.config.experiment_name)
            logger.info(f"MLflow experiment set: {self.config.experiment_name}")
            
        except Exception as e:
            logger.error(f"Failed to setup MLflow experiment: {e}")
            raise
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> str:
        """Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            tags: Optional tags for the run
            
        Returns:
            Run ID
        """
        # Generate run name if not provided
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"promptxpert_run_{timestamp}"
        
        # Default tags
        default_tags = {
            "framework": "dspy",
            "application": "prompt_optimization",
            "version": "1.0.0"
        }
        
        if tags:
            default_tags.update(tags)
        
        # Start run
        run = mlflow.start_run(run_name=run_name, tags=default_tags)
        self.run_id = run.info.run_id
        
        logger.info(f"Started MLflow run: {run_name} (ID: {self.run_id})")
        return self.run_id
    
    def log_config(self, config: PromptXpertConfig):
        """Log configuration parameters.
        
        Args:
            config: PromptXpert configuration
        """
        try:
            # Model configuration
            mlflow.log_param("primary_model", config.model.primary_model)
            mlflow.log_param("judge_model", config.model.judge_model)
            mlflow.log_param("primary_temperature", config.model.primary_temperature)
            mlflow.log_param("judge_temperature", config.model.judge_temperature)
            
            # Optimization configuration
            mlflow.log_param("max_labeled_demos", config.optimization.max_labeled_demos)
            mlflow.log_param("max_bootstrapped_demos", config.optimization.max_bootstrapped_demos)
            mlflow.log_param("auto_mode", config.optimization.auto_mode)
            mlflow.log_param("num_trials", config.optimization.num_trials)
            mlflow.log_param("minibatch_size", config.optimization.minibatch_size)
            
            # Data configuration
            mlflow.log_param("train_ratio", config.data.train_ratio)
            mlflow.log_param("val_ratio", config.data.val_ratio)
            mlflow.log_param("test_ratio", config.data.test_ratio)
            mlflow.log_param("random_seed", config.data.random_seed)
            
            # Metrics configuration
            mlflow.log_param("clarity_weight", config.metrics.clarity_weight)
            mlflow.log_param("specificity_weight", config.metrics.specificity_weight)
            mlflow.log_param("brevity_weight", config.metrics.brevity_weight)
            mlflow.log_param("safety_weight", config.metrics.safety_weight)
            
            # Global settings
            mlflow.log_param("cache_enabled", config.cache_enabled)
            mlflow.log_param("log_level", config.log_level)
            
            logger.info("Logged configuration parameters to MLflow")
            
        except Exception as e:
            logger.error(f"Failed to log config to MLflow: {e}")
    
    def log_dataset_info(self, train_size: int, val_size: int, test_size: int, 
                        dataset_stats: Optional[Dict[str, Any]] = None):
        """Log dataset information.
        
        Args:
            train_size: Size of training set
            val_size: Size of validation set
            test_size: Size of test set
            dataset_stats: Additional dataset statistics
        """
        try:
            mlflow.log_param("train_size", train_size)
            mlflow.log_param("val_size", val_size)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("total_size", train_size + val_size + test_size)
            
            if dataset_stats:
                for key, value in dataset_stats.items():
                    if isinstance(value, (int, float, str, bool)):
                        mlflow.log_param(f"dataset_{key}", value)
            
            logger.info("Logged dataset information to MLflow")
            
        except Exception as e:
            logger.error(f"Failed to log dataset info to MLflow: {e}")
    
    def log_optimization_progress(self, step: int, score: float, trial_info: Optional[Dict] = None):
        """Log optimization progress during training.
        
        Args:
            step: Optimization step/trial number
            score: Current best score
            trial_info: Additional trial information
        """
        try:
            mlflow.log_metric("optimization_score", score, step=step)
            
            if trial_info:
                for key, value in trial_info.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"trial_{key}", value, step=step)
            
        except Exception as e:
            logger.error(f"Failed to log optimization progress: {e}")
    
    def log_evaluation_results(self, results: Dict[str, Any], prefix: str = ""):
        """Log evaluation results.
        
        Args:
            results: Evaluation results dictionary
            prefix: Prefix for metric names
        """
        try:
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    metric_name = f"{prefix}_{key}" if prefix else key
                    mlflow.log_metric(metric_name, value)
            
            logger.info(f"Logged evaluation results to MLflow: {prefix}")
            
        except Exception as e:
            logger.error(f"Failed to log evaluation results: {e}")
    
    def log_model(self, model: dspy.Module, model_name: str = "promptxpert_model"):
        """Log the trained DSPy model.
        
        Args:
            model: Trained DSPy model
            model_name: Name for the logged model
        """
        try:
            # Save model temporarily
            temp_path = f"/tmp/{model_name}_{int(time.time())}"
            model.save(temp_path)
            
            # Log as artifact
            mlflow.log_artifacts(temp_path, artifact_path=model_name)
            
            # Also try to log with DSPy flavor if available
            try:
                mlflow.dspy.log_model(model, model_name)
            except AttributeError:
                # DSPy MLflow integration might not be available
                logger.warning("DSPy MLflow integration not available, logged as artifact only")
            
            logger.info(f"Logged model {model_name} to MLflow")
            
        except Exception as e:
            logger.error(f"Failed to log model to MLflow: {e}")
    
    def log_examples(self, examples: List[Dict[str, Any]], artifact_name: str = "examples"):
        """Log example data as artifact.
        
        Args:
            examples: List of example dictionaries
            artifact_name: Name for the artifact
        """
        try:
            # Save examples to JSON
            temp_file = f"/tmp/{artifact_name}_{int(time.time())}.json"
            with open(temp_file, 'w') as f:
                json.dump(examples, f, indent=2)
            
            mlflow.log_artifact(temp_file, artifact_path="examples")
            
            # Cleanup
            Path(temp_file).unlink()
            
            logger.info(f"Logged {len(examples)} examples to MLflow")
            
        except Exception as e:
            logger.error(f"Failed to log examples to MLflow: {e}")
    
    def log_comparison_results(self, baseline_score: float, optimized_score: float, 
                              improvement: float, details: Optional[Dict] = None):
        """Log baseline comparison results.
        
        Args:
            baseline_score: Baseline model score
            optimized_score: Optimized model score
            improvement: Absolute improvement
            details: Additional comparison details
        """
        try:
            mlflow.log_metric("baseline_score", baseline_score)
            mlflow.log_metric("optimized_score", optimized_score)
            mlflow.log_metric("absolute_improvement", improvement)
            
            if baseline_score > 0:
                relative_improvement = (improvement / baseline_score) * 100
                mlflow.log_metric("relative_improvement_percent", relative_improvement)
            
            if details:
                for key, value in details.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"comparison_{key}", value)
            
            logger.info("Logged comparison results to MLflow")
            
        except Exception as e:
            logger.error(f"Failed to log comparison results: {e}")
    
    def end_run(self, status: str = "FINISHED"):
        """End the current MLflow run.
        
        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        try:
            mlflow.end_run(status=status)
            logger.info(f"Ended MLflow run with status: {status}")
            
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {e}")
    
    def get_best_run(self, metric_name: str = "optimized_score") -> Optional[mlflow.entities.Run]:
        """Get the best run from the current experiment.
        
        Args:
            metric_name: Metric to optimize for
            
        Returns:
            Best run or None if no runs found
        """
        try:
            experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
            if experiment is None:
                return None
            
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[f"metrics.{metric_name} DESC"],
                max_results=1
            )
            
            if len(runs) > 0:
                return runs.iloc[0]
            return None
            
        except Exception as e:
            logger.error(f"Failed to get best run: {e}")
            return None


def setup_mlflow_experiment(config: PromptXpertConfig) -> MLflowTracker:
    """Setup MLflow experiment tracking.
    
    Args:
        config: PromptXpert configuration
        
    Returns:
        Configured MLflow tracker
    """
    tracker = MLflowTracker(config.mlflow)
    return tracker


def log_environment_info(tracker: MLflowTracker):
    """Log environment and system information.
    
    Args:
        tracker: MLflow tracker instance
    """
    try:
        import platform
        import dspy
        
        # System info
        mlflow.log_param("python_version", platform.python_version())
        mlflow.log_param("platform", platform.platform())
        mlflow.log_param("dspy_version", getattr(dspy, '__version__', 'unknown'))
        
        # Timestamp
        mlflow.log_param("experiment_timestamp", datetime.now().isoformat())
        
        logger.info("Logged environment information")
        
    except Exception as e:
        logger.error(f"Failed to log environment info: {e}")


class MLflowContextManager:
    """Context manager for MLflow runs."""
    
    def __init__(self, tracker: MLflowTracker, run_name: Optional[str] = None, 
                 tags: Optional[Dict[str, str]] = None):
        """Initialize context manager.
        
        Args:
            tracker: MLflow tracker
            run_name: Optional run name
            tags: Optional tags
        """
        self.tracker = tracker
        self.run_name = run_name
        self.tags = tags
        self.run_id = None
    
    def __enter__(self):
        """Start MLflow run."""
        self.run_id = self.tracker.start_run(self.run_name, self.tags)
        return self.tracker
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End MLflow run."""
        if exc_type is not None:
            self.tracker.end_run("FAILED")
        else:
            self.tracker.end_run("FINISHED")