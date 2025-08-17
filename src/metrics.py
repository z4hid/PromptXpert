"""Metrics for evaluating prompt optimization quality."""

import re
import dspy
import logging
from typing import Optional, Dict, Any, Union
from .signature import MultiCriteriaAssessment, PromptQualityAssessment
from .config import MetricsConfig

logger = logging.getLogger(__name__)


class PromptMetrics:
    """Class for evaluating prompt optimization quality with multiple metrics."""
    
    def __init__(self, judge_lm: dspy.LM, config: MetricsConfig):
        """Initialize metrics with judge language model and configuration.
        
        Args:
            judge_lm: Language model for judging quality
            config: Metrics configuration with weights
        """
        self.judge_lm = judge_lm
        self.config = config
        self.multi_criteria_judge = dspy.ChainOfThought(MultiCriteriaAssessment, lm=judge_lm)
        self.legacy_judge = dspy.ChainOfThought(PromptQualityAssessment, lm=judge_lm)
    
    def _parse_score(self, text: str, field_name: str) -> float:
        """Parse numeric score from text with robust error handling.
        
        Args:
            text: Text containing the score
            field_name: Name of the field for logging
            
        Returns:
            Parsed score clamped to [0, 1]
        """
        try:
            # Try to extract float between 0 and 1
            matches = re.findall(r'[0-1](?:\.\d+)?', str(text))
            if matches:
                score = float(matches[0])
                return max(0.0, min(1.0, score))
            
            # Try to extract any float and normalize
            matches = re.findall(r'\d+\.?\d*', str(text))
            if matches:
                score = float(matches[0])
                if score > 1.0:
                    score = score / 10.0 if score <= 10.0 else score / 100.0
                return max(0.0, min(1.0, score))
            
            logger.warning(f"Could not parse {field_name} score from: {text}")
            return 0.5  # Default to neutral
            
        except Exception as e:
            logger.warning(f"Error parsing {field_name} score: {e}")
            return 0.5
    
    def multi_criteria_metric(self, example: dspy.Example, prediction: Union[dspy.Prediction, str], trace=None) -> float:
        """Multi-criteria metric using weighted combination of quality dimensions.
        
        Args:
            example: Example with initial_prompt
            prediction: Prediction with optimized_prompt or string
            trace: Optional trace for debugging
            
        Returns:
            Weighted composite score [0, 1]
        """
        try:
            # Extract optimized prompt from prediction
            if isinstance(prediction, str):
                optimized_prompt = prediction
            elif hasattr(prediction, 'optimized_prompt'):
                optimized_prompt = prediction.optimized_prompt
            else:
                optimized_prompt = str(prediction)
            
            # Get multi-criteria assessment
            assessment = self.multi_criteria_judge(
                initial_prompt=example.initial_prompt,
                optimized_prompt=optimized_prompt
            )
            
            # Parse individual scores
            clarity = self._parse_score(getattr(assessment, 'clarity', 0.5), 'clarity')
            specificity = self._parse_score(getattr(assessment, 'specificity', 0.5), 'specificity')
            brevity = self._parse_score(getattr(assessment, 'brevity', 0.5), 'brevity')
            safety = self._parse_score(getattr(assessment, 'safety', 0.5), 'safety')
            
            # Calculate weighted score
            score = (
                clarity * self.config.clarity_weight +
                specificity * self.config.specificity_weight +
                brevity * self.config.brevity_weight +
                safety * self.config.safety_weight
            )
            
            # Store detailed scores for analysis
            if trace is not None:
                trace.append({
                    'type': 'multi_criteria_scores',
                    'clarity': clarity,
                    'specificity': specificity,
                    'brevity': brevity,
                    'safety': safety,
                    'composite': score,
                    'reasoning': getattr(assessment, 'reasoning', '')
                })
            
            return float(score)
            
        except Exception as e:
            logger.error(f"Error in multi-criteria metric: {e}")
            return 0.5
    
    def legacy_metric(self, example: dspy.Example, prediction: Union[dspy.Prediction, str], trace=None) -> float:
        """Legacy single-score metric for backward compatibility.
        
        Args:
            example: Example with initial_prompt
            prediction: Prediction with optimized_prompt or string
            trace: Optional trace for debugging
            
        Returns:
            Single quality score [0, 1]
        """
        try:
            # Extract optimized prompt from prediction
            if isinstance(prediction, str):
                optimized_prompt = prediction
            elif hasattr(prediction, 'optimized_prompt'):
                optimized_prompt = prediction.optimized_prompt
            else:
                optimized_prompt = str(prediction)
            
            # Get legacy assessment
            assessment = self.legacy_judge(
                initial_prompt=example.initial_prompt,
                optimized_prompt=optimized_prompt
            )
            
            # Parse rating
            rating = self._parse_score(getattr(assessment, 'rating', 0.5), 'rating')
            
            # Store reasoning for analysis
            if trace is not None:
                trace.append({
                    'type': 'legacy_score',
                    'rating': rating,
                    'reasoning': getattr(assessment, 'reasoning', '')
                })
            
            return float(rating)
            
        except Exception as e:
            logger.error(f"Error in legacy metric: {e}")
            return 0.5
    
    def heuristic_metric(self, example: dspy.Example, prediction: Union[dspy.Prediction, str], trace=None) -> float:
        """Simple heuristic metric for sanity checking.
        
        Args:
            example: Example with initial_prompt
            prediction: Prediction with optimized_prompt or string
            trace: Optional trace for debugging
            
        Returns:
            Heuristic quality score [0, 1]
        """
        try:
            # Extract optimized prompt from prediction
            if isinstance(prediction, str):
                optimized_prompt = prediction
            elif hasattr(prediction, 'optimized_prompt'):
                optimized_prompt = prediction.optimized_prompt
            else:
                optimized_prompt = str(prediction)
            
            initial_prompt = example.initial_prompt
            
            # Length improvement (up to 0.3 points)
            initial_len = len(initial_prompt.split())
            optimized_len = len(optimized_prompt.split())
            length_score = min(0.3, max(0, (optimized_len - initial_len) / max(initial_len, 1) * 0.3))
            
            # Specificity keywords (up to 0.3 points)
            specificity_keywords = [
                'specific', 'detailed', 'clear', 'comprehensive', 'include', 'ensure',
                'provide', 'format', 'structure', 'requirements', 'criteria', 'examples'
            ]
            keyword_count = sum(1 for word in specificity_keywords 
                              if word in optimized_prompt.lower() and word not in initial_prompt.lower())
            keyword_score = min(0.3, keyword_count * 0.05)
            
            # Question/instruction words (up to 0.2 points)
            instruction_patterns = ['how', 'what', 'why', 'when', 'where', 'which', 'please', 'should']
            instruction_count = sum(1 for pattern in instruction_patterns 
                                  if pattern in optimized_prompt.lower())
            instruction_score = min(0.2, instruction_count * 0.03)
            
            # Penalty for excessive length (>500 words)
            length_penalty = max(0, (optimized_len - 500) * 0.001) if optimized_len > 500 else 0
            
            # Base score for any non-trivial change
            base_score = 0.2 if optimized_prompt != initial_prompt else 0
            
            # Composite heuristic score
            heuristic_score = base_score + length_score + keyword_score + instruction_score - length_penalty
            heuristic_score = max(0.0, min(1.0, heuristic_score))
            
            # Store details for analysis
            if trace is not None:
                trace.append({
                    'type': 'heuristic_score',
                    'base_score': base_score,
                    'length_score': length_score,
                    'keyword_score': keyword_score,
                    'instruction_score': instruction_score,
                    'length_penalty': length_penalty,
                    'total': heuristic_score
                })
            
            return float(heuristic_score)
            
        except Exception as e:
            logger.error(f"Error in heuristic metric: {e}")
            return 0.5
    
    def composite_metric(self, example: dspy.Example, prediction: Union[dspy.Prediction, str], trace=None) -> float:
        """Composite metric combining multiple scoring approaches.
        
        Args:
            example: Example with initial_prompt
            prediction: Prediction with optimized_prompt or string
            trace: Optional trace for debugging
            
        Returns:
            Composite score combining multiple metrics
        """
        # Get all metric scores
        multi_score = self.multi_criteria_metric(example, prediction, trace)
        heuristic_score = self.heuristic_metric(example, prediction, trace)
        
        # Weighted combination (favor LLM judge but use heuristic as sanity check)
        composite = 0.8 * multi_score + 0.2 * heuristic_score
        
        return float(composite)


def create_metric_function(judge_lm: dspy.LM, config: MetricsConfig, metric_type: str = "multi_criteria"):
    """Factory function to create metric functions for DSPy evaluation.
    
    Args:
        judge_lm: Judge language model
        config: Metrics configuration
        metric_type: Type of metric ('multi_criteria', 'legacy', 'heuristic', 'composite')
        
    Returns:
        Metric function compatible with DSPy Evaluate
    """
    metrics = PromptMetrics(judge_lm, config)
    
    if metric_type == "multi_criteria":
        return metrics.multi_criteria_metric
    elif metric_type == "legacy":
        return metrics.legacy_metric
    elif metric_type == "heuristic":
        return metrics.heuristic_metric
    elif metric_type == "composite":
        return metrics.composite_metric
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")


def evaluate_with_baseline(
    baseline_program: dspy.Module,
    optimized_program: dspy.Module,
    examples: list,
    metric_function,
    display_progress: bool = True
) -> Dict[str, Any]:
    """Evaluate both baseline and optimized programs for comparison.
    
    Args:
        baseline_program: Baseline program (e.g., identity transform)
        optimized_program: Optimized program
        examples: Test examples
        metric_function: Metric function to use
        display_progress: Whether to display progress
        
    Returns:
        Dictionary with comparative results
    """
    from dspy.evaluate import Evaluate
    
    # Evaluate baseline
    baseline_evaluator = Evaluate(
        devset=examples,
        metric=metric_function,
        display_table=False,
        display_progress=display_progress
    )
    baseline_results = baseline_evaluator(baseline_program)
    
    # Evaluate optimized
    optimized_evaluator = Evaluate(
        devset=examples,
        metric=metric_function,
        display_table=False,
        display_progress=display_progress
    )
    optimized_results = optimized_evaluator(optimized_program)
    
    # Calculate improvement
    baseline_score = baseline_results.get('score', 0) if isinstance(baseline_results, dict) else baseline_results
    optimized_score = optimized_results.get('score', 0) if isinstance(optimized_results, dict) else optimized_results
    
    improvement = optimized_score - baseline_score
    relative_improvement = (improvement / baseline_score * 100) if baseline_score > 0 else 0
    
    return {
        'baseline_score': baseline_score,
        'optimized_score': optimized_score,
        'absolute_improvement': improvement,
        'relative_improvement_percent': relative_improvement,
        'baseline_results': baseline_results,
        'optimized_results': optimized_results
    }