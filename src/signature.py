"""DSPy Signatures for PromptXpert."""

import dspy
from typing import List


class PromptOptimization(dspy.Signature):
    """Signature for optimizing prompts with structured output."""
    
    initial_prompt: str = dspy.InputField(
        desc="The original user prompt that needs optimization"
    )
    
    objective: str = dspy.OutputField(
        desc="One-sentence task objective that clearly states what the prompt aims to achieve"
    )
    
    constraints: str = dspy.OutputField(
        desc="List of 3-6 concrete constraints and requirements for the task, formatted as bullet points"
    )
    
    format_instructions: str = dspy.OutputField(
        desc="Explicit response formatting instructions including structure, length, and style requirements"
    )
    
    optimized_prompt: str = dspy.OutputField(
        desc="Final optimized prompt that combines objective, constraints, and format instructions into a clear, comprehensive, and actionable instruction"
    )


class MultiCriteriaAssessment(dspy.Signature):
    """Signature for multi-criteria prompt quality assessment."""
    
    initial_prompt: str = dspy.InputField(
        desc="The original prompt before optimization"
    )
    
    optimized_prompt: str = dspy.InputField(
        desc="The optimized version of the prompt"
    )
    
    clarity: float = dspy.OutputField(
        desc="Clarity score (0.0-1.0): How clear and unambiguous is the optimized prompt compared to the original? 1.0 = perfectly clear, 0.0 = very unclear"
    )
    
    specificity: float = dspy.OutputField(
        desc="Specificity score (0.0-1.0): How specific and detailed are the instructions? 1.0 = very specific with concrete details, 0.0 = too vague or generic"
    )
    
    brevity: float = dspy.OutputField(
        desc="Brevity score (0.0-1.0): Is the prompt appropriately concise without unnecessary verbosity? 1.0 = optimal length, 0.0 = too verbose or too brief"
    )
    
    safety: float = dspy.OutputField(
        desc="Safety score (0.0-1.0): Does the prompt maintain safety and ethical guidelines? 1.0 = completely safe, 0.0 = potential safety concerns"
    )
    
    reasoning: str = dspy.OutputField(
        desc="Detailed explanation of the scoring rationale, highlighting specific improvements or concerns in each criteria"
    )


class PromptQualityAssessment(dspy.Signature):
    """Legacy signature for simple quality assessment (backward compatibility)."""
    
    initial_prompt: str = dspy.InputField(
        desc="The original user prompt"
    )
    
    optimized_prompt: str = dspy.InputField(
        desc="The optimized version of the prompt"
    )
    
    rating: float = dspy.OutputField(
        desc="Overall quality improvement rating from 0.0 to 1.0, where 0.5 indicates no improvement, above 0.5 indicates improvement, below 0.5 indicates degradation"
    )
    
    reasoning: str = dspy.OutputField(
        desc="Detailed explanation of why the optimized prompt is better, worse, or equivalent to the original"
    )


class PromptValidation(dspy.Signature):
    """Signature for validating prompt completeness and quality."""
    
    prompt: str = dspy.InputField(
        desc="The prompt to validate"
    )
    
    completeness: float = dspy.OutputField(
        desc="Completeness score (0.0-1.0): Does the prompt contain all necessary information for the task?"
    )
    
    ambiguity: float = dspy.OutputField(
        desc="Ambiguity score (0.0-1.0): How ambiguous is the prompt? 0.0 = very ambiguous, 1.0 = completely unambiguous"
    )
    
    actionability: float = dspy.OutputField(
        desc="Actionability score (0.0-1.0): How actionable and executable are the instructions?"
    )
    
    needs_refinement: bool = dspy.OutputField(
        desc="Boolean indicating whether the prompt needs further refinement"
    )
    
    suggestions: str = dspy.OutputField(
        desc="Specific suggestions for improvement if refinement is needed"
    )
