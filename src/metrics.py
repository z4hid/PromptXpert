import dspy
from typing import Optional, Dict, Any
from tenacity import RetryError
from .logging_setup import logger

class PromptQualityAssessment(dspy.Signature):
    initial_prompt: str = dspy.InputField(desc="Original prompt.")
    optimized_prompt: str = dspy.InputField(desc="Optimized prompt.")
    clarity: float = dspy.OutputField(desc="How clear is the optimized prompt vs initial? [0.0-1.0]")
    specificity: float = dspy.OutputField(desc="How specific/actionable is the optimized prompt? [0.0-1.0]")
    completeness: float = dspy.OutputField(desc="Does it include necessary info/constraints? [0.0-1.0]")
    effectiveness: float = dspy.OutputField(desc="Likelihood to yield desired outputs? [0.0-1.0]")
    reasoning: str = dspy.OutputField(desc="Short explanation of the scores.")

class MultiCriteriaPromptMetric:
    def __init__(self, judge_lm: dspy.LM, weights: Optional[Dict[str, float]] = None):
        self.judge = dspy.ChainOfThought(PromptQualityAssessment, lm=judge_lm)
        self.weights = weights or {"clarity": 0.25, "specificity": 0.25, "completeness": 0.25, "effectiveness": 0.25}

    def __call__(self, example: dspy.Example, pred: Any, trace: Optional[Any] = None) -> float:
        if not isinstance(pred, dspy.Prediction):
            optimized_prompt = str(pred)
        else:
            optimized_prompt = getattr(pred, "optimized_prompt", str(pred))
        try:
            assess = self.judge(initial_prompt=example.initial_prompt, optimized_prompt=optimized_prompt)
            scores = {
                "clarity": float(assess.clarity),
                "specificity": float(assess.specificity),
                "completeness": float(assess.completeness),
                "effectiveness": float(assess.effectiveness),
            }
            for k, v in scores.items():
                if v < 0.0 or v > 1.0:
                    scores[k] = max(0.0, min(1.0, v))
            return float(sum(self.weights[k] * scores[k] for k in self.weights))
        except RetryError as e:
            logger.error(f"Judge failed due to RetryError for example: {example.initial_prompt}. Error: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Judge failed: {e}")
            return 0.0
