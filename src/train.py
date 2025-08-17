"""Training pipeline for PromptXpert with MLflow tracking."""

import os
import logging
import dspy
from dspy.teleprompt import MIPROv2
from dspy.evaluate import Evaluate
from pathlib import Path
from typing import Optional, Dict, Any

from .config import PromptXpertConfig, load_config_from_env
from .utils import prepare_dataset, set_reproducibility_seeds, validate_dataset_quality
from .module import PromptXpertProgram, BaselineProgram
from .metrics import create_metric_function, evaluate_with_baseline
from .mlflow_integration import setup_mlflow_experiment, MLflowContextManager, log_environment_info

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_language_models(config: PromptXpertConfig) -> tuple[dspy.LM, dspy.LM]:
    """Setup primary and judge language models.
    
    Args:
        config: Configuration with model settings
        
    Returns:
        Tuple of (primary_lm, judge_lm)
    """
    # Check for API key
    api_key = os.getenv(config.model.api_key_env)
    if not api_key:
        raise ValueError(f"API key not found in environment variable: {config.model.api_key_env}")
    
    # Setup primary model (for optimization tasks)
    primary_lm = dspy.Groq(
        model=config.model.primary_model,
        api_key=api_key,
        temperature=config.model.primary_temperature,
        max_tokens=config.model.max_tokens
    )
    
    # Setup judge model (for evaluation, different model to reduce bias)
    judge_lm = dspy.Groq(
        model=config.model.judge_model,
        api_key=api_key,
        temperature=config.model.judge_temperature,
        max_tokens=config.model.max_tokens
    )
    
    # Configure DSPy to use primary model by default
    dspy.settings.configure(lm=primary_lm)
    
    logger.info(f"Setup models - Primary: {config.model.primary_model}, Judge: {config.model.judge_model}")
    return primary_lm, judge_lm


def train_prompt_optimizer(config: PromptXpertConfig) -> Dict[str, Any]:
    """Main training function for prompt optimization.
    
    Args:
        config: PromptXpert configuration
        
    Returns:
        Dictionary with training results and paths
    """
    # Setup MLflow tracking
    tracker = setup_mlflow_experiment(config)
    
    with MLflowContextManager(tracker, "promptxpert_training") as mlflow_tracker:
        try:
            # Log configuration and environment
            mlflow_tracker.log_config(config)
            log_environment_info(mlflow_tracker)
            
            # Set reproducibility
            set_reproducibility_seeds(config.random_seed)
            
            # Setup language models
            primary_lm, judge_lm = setup_language_models(config)
            
            # Prepare dataset
            logger.info("Preparing dataset...")
            train_examples, val_examples, test_examples = prepare_dataset(config)
            
            # Validate dataset quality
            dataset_stats = validate_dataset_quality(train_examples + val_examples + test_examples)
            logger.info(f"Dataset quality stats: {dataset_stats}")
            
            # Log dataset information
            mlflow_tracker.log_dataset_info(
                len(train_examples), len(val_examples), len(test_examples), dataset_stats
            )
            
            # Create metric function
            metric_function = create_metric_function(judge_lm, config.metrics, "multi_criteria")
            
            # Initialize programs
            logger.info("Initializing programs...")
            uncompiled_program = PromptXpertProgram()
            baseline_program = BaselineProgram()
            
            # Setup optimizer
            logger.info("Setting up MIPROv2 optimizer...")
            optimizer = MIPROv2(
                metric=metric_function,
                max_bootstrapped_demos=config.optimization.max_bootstrapped_demos,
                max_labeled_demos=config.optimization.max_labeled_demos,
                auto=config.optimization.auto_mode,
                num_trials=config.optimization.num_trials
            )
            
            # Compile (optimize) the program
            logger.info("Starting compilation/optimization...")
            compiled_program = optimizer.compile(
                student=uncompiled_program,
                trainset=train_examples,
                valset=val_examples,
                minibatch_size=config.optimization.minibatch_size
            )
            
            # Evaluate baseline vs optimized
            logger.info("Evaluating baseline vs optimized programs...")
            comparison_results = evaluate_with_baseline(
                baseline_program=baseline_program,
                optimized_program=compiled_program,
                examples=test_examples,
                metric_function=metric_function,
                display_progress=True
            )
            
            # Log evaluation results
            mlflow_tracker.log_comparison_results(
                baseline_score=comparison_results['baseline_score'],
                optimized_score=comparison_results['optimized_score'],
                improvement=comparison_results['absolute_improvement'],
                details={
                    'relative_improvement_percent': comparison_results['relative_improvement_percent']
                }
            )
            
            # Additional metrics with different metric types
            logger.info("Evaluating with different metric types...")
            
            # Heuristic metric
            heuristic_metric = create_metric_function(judge_lm, config.metrics, "heuristic")
            heuristic_eval = Evaluate(devset=test_examples, metric=heuristic_metric, display_table=False)
            heuristic_score = heuristic_eval(compiled_program)
            mlflow_tracker.log_evaluation_results({"heuristic_score": heuristic_score}, "additional")
            
            # Legacy metric
            legacy_metric = create_metric_function(judge_lm, config.metrics, "legacy")
            legacy_eval = Evaluate(devset=test_examples, metric=legacy_metric, display_table=False)
            legacy_score = legacy_eval(compiled_program)
            mlflow_tracker.log_evaluation_results({"legacy_score": legacy_score}, "additional")
            
            # Save compiled program
            output_dir = Path(config.output_dir)
            output_dir.mkdir(exist_ok=True)
            
            model_path = output_dir / "compiled_promptxpert_model"
            compiled_program.save(str(model_path))
            logger.info(f"Saved compiled program to {model_path}")
            
            # Log model to MLflow
            mlflow_tracker.log_model(compiled_program, "promptxpert_optimized")
            
            # Log some example predictions
            logger.info("Logging example predictions...")
            example_predictions = []
            for i, example in enumerate(test_examples[:5]):  # Log first 5 examples
                try:
                    baseline_pred = baseline_program(initial_prompt=example.initial_prompt)
                    optimized_pred = compiled_program(initial_prompt=example.initial_prompt)
                    
                    example_predictions.append({
                        'example_id': i,
                        'initial_prompt': example.initial_prompt,
                        'expected_optimized': example.optimized_prompt,
                        'baseline_prediction': getattr(baseline_pred, 'optimized_prompt', str(baseline_pred)),
                        'optimized_prediction': getattr(optimized_pred, 'optimized_prompt', str(optimized_pred))
                    })
                except Exception as e:
                    logger.warning(f"Failed to generate prediction for example {i}: {e}")
            
            mlflow_tracker.log_examples(example_predictions, "prediction_examples")
            
            # Prepare results
            results = {
                'model_path': str(model_path),
                'baseline_score': comparison_results['baseline_score'],
                'optimized_score': comparison_results['optimized_score'],
                'improvement': comparison_results['absolute_improvement'],
                'relative_improvement_percent': comparison_results['relative_improvement_percent'],
                'dataset_stats': dataset_stats,
                'config': config,
                'train_size': len(train_examples),
                'val_size': len(val_examples),
                'test_size': len(test_examples),
                'heuristic_score': heuristic_score,
                'legacy_score': legacy_score
            }
            
            logger.info(f"Training completed successfully!")
            logger.info(f"Baseline score: {comparison_results['baseline_score']:.3f}")
            logger.info(f"Optimized score: {comparison_results['optimized_score']:.3f}")
            logger.info(f"Improvement: {comparison_results['absolute_improvement']:.3f} "
                       f"({comparison_results['relative_improvement_percent']:.1f}%)")
            
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise


def main():
    """Main training entry point."""
    try:
        # Load configuration
        config = load_config_from_env()
        
        # Print configuration summary
        logger.info("=== PromptXpert Training Configuration ===")
        logger.info(f"Primary Model: {config.model.primary_model}")
        logger.info(f"Judge Model: {config.model.judge_model}")
        logger.info(f"Optimization Mode: {config.optimization.auto_mode}")
        logger.info(f"Max Trials: {config.optimization.num_trials}")
        logger.info(f"Dataset: {config.data.dataset_path}")
        logger.info(f"Output Directory: {config.output_dir}")
        logger.info("=" * 45)
        
        # Run training
        results = train_prompt_optimizer(config)
        
        # Print summary
        print("\n" + "=" * 50)
        print("TRAINING SUMMARY")
        print("=" * 50)
        print(f"Model saved to: {results['model_path']}")
        print(f"Baseline score: {results['baseline_score']:.3f}")
        print(f"Optimized score: {results['optimized_score']:.3f}")
        print(f"Absolute improvement: {results['improvement']:.3f}")
        print(f"Relative improvement: {results['relative_improvement_percent']:.1f}%")
        print(f"Dataset size: {results['train_size']} train, {results['val_size']} val, {results['test_size']} test")
        print("=" * 50)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()
