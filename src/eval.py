"""Evaluation pipeline for PromptXpert with comprehensive reporting."""

import os
import logging
import json
import dspy
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List
from dspy.evaluate import Evaluate

from .config import PromptXpertConfig, load_config_from_env
from .utils import prepare_dataset, set_reproducibility_seeds
from .module import PromptXpertProgram, BaselineProgram
from .metrics import create_metric_function, evaluate_with_baseline, PromptMetrics
from .mlflow_integration import setup_mlflow_experiment, MLflowContextManager, log_environment_info

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_trained_model(model_path: str) -> PromptXpertProgram:
    """Load a trained PromptXpert model.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded model
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = PromptXpertProgram()
    model.load(model_path)
    logger.info(f"Loaded model from {model_path}")
    return model


def evaluate_model_comprehensive(
    model: PromptXpertProgram,
    test_examples: List[dspy.Example],
    config: PromptXpertConfig,
    judge_lm: dspy.LM
) -> Dict[str, Any]:
    """Comprehensive evaluation with multiple metrics.
    
    Args:
        model: Model to evaluate
        test_examples: Test examples
        config: Configuration
        judge_lm: Judge language model
        
    Returns:
        Comprehensive evaluation results
    """
    results = {}
    
    # Initialize metrics
    metrics = PromptMetrics(judge_lm, config.metrics)
    
    # Evaluate with different metric types
    metric_types = ["multi_criteria", "legacy", "heuristic", "composite"]
    
    for metric_type in metric_types:
        logger.info(f"Evaluating with {metric_type} metric...")
        
        metric_function = create_metric_function(judge_lm, config.metrics, metric_type)
        evaluator = Evaluate(
            devset=test_examples,
            metric=metric_function,
            display_table=False,
            display_progress=True
        )
        
        score = evaluator(model)
        results[f"{metric_type}_score"] = score
        
        logger.info(f"{metric_type.title()} score: {score:.3f}")
    
    # Detailed analysis on a subset of examples
    logger.info("Performing detailed analysis...")
    detailed_results = []
    
    for i, example in enumerate(test_examples[:10]):  # Analyze first 10 examples
        try:
            prediction = model(initial_prompt=example.initial_prompt)
            
            # Get detailed scores
            trace = []
            multi_score = metrics.multi_criteria_metric(example, prediction, trace)
            heuristic_score = metrics.heuristic_metric(example, prediction, trace)
            
            detailed_result = {
                'example_id': i,
                'initial_prompt': example.initial_prompt,
                'expected_optimized': example.optimized_prompt,
                'predicted_optimized': getattr(prediction, 'optimized_prompt', str(prediction)),
                'multi_criteria_score': multi_score,
                'heuristic_score': heuristic_score,
                'trace': trace
            }
            
            # Extract detailed criteria scores if available
            if trace:
                for entry in trace:
                    if entry.get('type') == 'multi_criteria_scores':
                        detailed_result.update({
                            'clarity_score': entry.get('clarity', 0),
                            'specificity_score': entry.get('specificity', 0),
                            'brevity_score': entry.get('brevity', 0),
                            'safety_score': entry.get('safety', 0),
                            'reasoning': entry.get('reasoning', '')
                        })
                        break
            
            detailed_results.append(detailed_result)
            
        except Exception as e:
            logger.warning(f"Failed to analyze example {i}: {e}")
    
    results['detailed_analysis'] = detailed_results
    
    # Calculate statistics
    if detailed_results:
        criteria_scores = ['clarity_score', 'specificity_score', 'brevity_score', 'safety_score']
        for criteria in criteria_scores:
            scores = [r.get(criteria, 0) for r in detailed_results if criteria in r]
            if scores:
                results[f"{criteria}_mean"] = sum(scores) / len(scores)
                results[f"{criteria}_std"] = (sum((x - results[f"{criteria}_mean"])**2 for x in scores) / len(scores))**0.5
    
    return results


def compare_with_baseline(
    optimized_model: PromptXpertProgram,
    test_examples: List[dspy.Example],
    config: PromptXpertConfig,
    judge_lm: dspy.LM
) -> Dict[str, Any]:
    """Compare optimized model with baseline.
    
    Args:
        optimized_model: Optimized model
        test_examples: Test examples
        config: Configuration
        judge_lm: Judge language model
        
    Returns:
        Comparison results
    """
    logger.info("Comparing with baseline...")
    
    baseline_model = BaselineProgram()
    metric_function = create_metric_function(judge_lm, config.metrics, "multi_criteria")
    
    comparison = evaluate_with_baseline(
        baseline_program=baseline_model,
        optimized_program=optimized_model,
        examples=test_examples,
        metric_function=metric_function,
        display_progress=True
    )
    
    # Detailed comparison on examples
    comparison_examples = []
    for i, example in enumerate(test_examples[:5]):
        try:
            baseline_pred = baseline_model(initial_prompt=example.initial_prompt)
            optimized_pred = optimized_model(initial_prompt=example.initial_prompt)
            
            # Calculate scores for both
            baseline_score = metric_function(example, baseline_pred)
            optimized_score = metric_function(example, optimized_pred)
            
            comparison_examples.append({
                'example_id': i,
                'initial_prompt': example.initial_prompt,
                'baseline_prediction': getattr(baseline_pred, 'optimized_prompt', str(baseline_pred)),
                'optimized_prediction': getattr(optimized_pred, 'optimized_prompt', str(optimized_pred)),
                'baseline_score': baseline_score,
                'optimized_score': optimized_score,
                'improvement': optimized_score - baseline_score
            })
            
        except Exception as e:
            logger.warning(f"Failed to compare example {i}: {e}")
    
    comparison['detailed_comparisons'] = comparison_examples
    return comparison


def save_evaluation_report(results: Dict[str, Any], output_path: str):
    """Save comprehensive evaluation report.
    
    Args:
        results: Evaluation results
        output_path: Path to save the report
    """
    # Create readable report
    report = {
        'evaluation_summary': {
            'multi_criteria_score': results.get('multi_criteria_score', 0),
            'legacy_score': results.get('legacy_score', 0),
            'heuristic_score': results.get('heuristic_score', 0),
            'composite_score': results.get('composite_score', 0),
        },
        'criteria_breakdown': {
            'clarity_mean': results.get('clarity_score_mean', 0),
            'specificity_mean': results.get('specificity_score_mean', 0),
            'brevity_mean': results.get('brevity_score_mean', 0),
            'safety_mean': results.get('safety_score_mean', 0),
        },
        'baseline_comparison': results.get('baseline_comparison', {}),
        'detailed_analysis': results.get('detailed_analysis', []),
    }
    
    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Saved evaluation report to {output_path}")


def evaluate_model_on_new_prompts(model: PromptXpertProgram, prompts: List[str]) -> List[Dict[str, Any]]:
    """Evaluate model on new prompts (inference mode).
    
    Args:
        model: Trained model
        prompts: List of prompts to optimize
        
    Returns:
        List of optimization results
    """
    results = []
    
    for i, prompt in enumerate(prompts):
        try:
            prediction = model(initial_prompt=prompt)
            
            result = {
                'id': i,
                'initial_prompt': prompt,
                'optimized_prompt': getattr(prediction, 'optimized_prompt', str(prediction)),
                'objective': getattr(prediction, 'objective', ''),
                'constraints': getattr(prediction, 'constraints', ''),
                'format_instructions': getattr(prediction, 'format_instructions', '')
            }
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Failed to optimize prompt {i}: {e}")
            results.append({
                'id': i,
                'initial_prompt': prompt,
                'error': str(e)
            })
    
    return results


def run_evaluation(config: PromptXpertConfig, model_path: Optional[str] = None) -> Dict[str, Any]:
    """Main evaluation function.
    
    Args:
        config: Configuration
        model_path: Optional path to trained model
        
    Returns:
        Evaluation results
    """
    # Setup MLflow tracking
    tracker = setup_mlflow_experiment(config)
    
    with MLflowContextManager(tracker, "promptxpert_evaluation") as mlflow_tracker:
        try:
            # Log configuration
            mlflow_tracker.log_config(config)
            log_environment_info(mlflow_tracker)
            
            # Set reproducibility
            set_reproducibility_seeds(config.random_seed)
            
            # Setup language models
            from .train import setup_language_models
            primary_lm, judge_lm = setup_language_models(config)
            
            # Load model
            if model_path is None:
                model_path = Path(config.output_dir) / "compiled_promptxpert_model"
            
            model = load_trained_model(model_path)
            
            # Prepare test dataset
            _, _, test_examples = prepare_dataset(config)
            
            logger.info(f"Evaluating model on {len(test_examples)} test examples...")
            
            # Comprehensive evaluation
            eval_results = evaluate_model_comprehensive(model, test_examples, config, judge_lm)
            
            # Baseline comparison
            comparison_results = compare_with_baseline(model, test_examples, config, judge_lm)
            
            # Combine results
            results = {
                **eval_results,
                'baseline_comparison': comparison_results,
                'model_path': str(model_path),
                'test_size': len(test_examples)
            }
            
            # Log to MLflow
            mlflow_tracker.log_evaluation_results(eval_results, "evaluation")
            mlflow_tracker.log_comparison_results(
                baseline_score=comparison_results['baseline_score'],
                optimized_score=comparison_results['optimized_score'],
                improvement=comparison_results['absolute_improvement']
            )
            
            # Save detailed report
            output_dir = Path(config.output_dir)
            report_path = output_dir / "evaluation_report.json"
            save_evaluation_report(results, str(report_path))
            
            logger.info("Evaluation completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise


def main():
    """Main evaluation entry point."""
    try:
        # Load configuration
        config = load_config_from_env()
        
        # Check for custom model path
        model_path = os.getenv("PROMPTXPERT_MODEL_PATH")
        
        logger.info("=== PromptXpert Evaluation ===")
        logger.info(f"Model path: {model_path or 'default'}")
        logger.info(f"Dataset: {config.data.dataset_path}")
        logger.info("=" * 35)
        
        # Run evaluation
        results = run_evaluation(config, model_path)
        
        # Print summary
        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Multi-criteria score: {results.get('multi_criteria_score', 0):.3f}")
        print(f"Legacy score: {results.get('legacy_score', 0):.3f}")
        print(f"Heuristic score: {results.get('heuristic_score', 0):.3f}")
        print(f"Composite score: {results.get('composite_score', 0):.3f}")
        
        if 'baseline_comparison' in results:
            comp = results['baseline_comparison']
            print(f"\nBaseline comparison:")
            print(f"  Baseline score: {comp.get('baseline_score', 0):.3f}")
            print(f"  Optimized score: {comp.get('optimized_score', 0):.3f}")
            print(f"  Improvement: {comp.get('absolute_improvement', 0):.3f}")
            print(f"  Relative improvement: {comp.get('relative_improvement_percent', 0):.1f}%")
        
        print(f"\nTest examples: {results.get('test_size', 0)}")
        print(f"Detailed report saved to: {config.output_dir}/evaluation_report.json")
        print("=" * 50)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == '__main__':
    main()
