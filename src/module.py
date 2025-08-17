"""Core DSPy Module for PromptXpert."""

import dspy
from typing import Optional, Dict, Any
from .signature import PromptOptimization, PromptValidation


class PromptXpertProgram(dspy.Module):
    """Main DSPy module for prompt optimization using Chain of Thought reasoning."""
    
    def __init__(self, use_validation: bool = False):
        """Initialize the PromptXpert program.
        
        Args:
            use_validation: Whether to include validation step for iterative refinement
        """
        super().__init__()
        self.optimizer = dspy.ChainOfThought(PromptOptimization)
        self.use_validation = use_validation
        
        if use_validation:
            self.validator = dspy.ChainOfThought(PromptValidation)
    
    def forward(self, initial_prompt: str) -> dspy.Prediction:
        """Forward pass through the optimization module.
        
        Args:
            initial_prompt: The original prompt to optimize
            
        Returns:
            Prediction containing optimized prompt and intermediate reasoning
        """
        # Primary optimization
        result = self.optimizer(initial_prompt=initial_prompt)
        
        # Optional validation and refinement
        if self.use_validation and hasattr(result, 'optimized_prompt'):
            validation = self.validator(prompt=result.optimized_prompt)
            
            # If validation suggests refinement is needed, we could implement
            # iterative refinement here (future enhancement)
            if hasattr(validation, 'needs_refinement') and validation.needs_refinement:
                # For now, just add validation info to the result
                result.validation_score = (
                    validation.completeness * 0.4 + 
                    validation.ambiguity * 0.3 + 
                    validation.actionability * 0.3
                )
                result.validation_suggestions = validation.suggestions
        
        return result


class BaselineProgram(dspy.Module):
    """Baseline program that returns the original prompt unchanged for comparison."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, initial_prompt: str) -> dspy.Prediction:
        """Return the original prompt as the 'optimized' version.
        
        Args:
            initial_prompt: The original prompt
            
        Returns:
            Prediction with original prompt as optimized_prompt
        """
        return dspy.Prediction(
            optimized_prompt=initial_prompt,
            objective="Baseline - no optimization applied",
            constraints="Original prompt constraints maintained",
            format_instructions="Original format maintained"
        )


class PromptRefiner(dspy.Module):
    """Module for iterative prompt refinement based on feedback."""
    
    def __init__(self):
        super().__init__()
        self.refiner = dspy.ChainOfThought(
            "feedback: str, current_prompt: str -> refined_prompt: str, improvements: str"
        )
    
    def forward(self, current_prompt: str, feedback: str) -> dspy.Prediction:
        """Refine a prompt based on feedback.
        
        Args:
            current_prompt: The current version of the prompt
            feedback: Feedback or suggestions for improvement
            
        Returns:
            Prediction with refined prompt
        """
        return self.refiner(feedback=feedback, current_prompt=current_prompt)


class PromptComposer(dspy.Module):
    """Module for composing final prompt from structured components."""
    
    def __init__(self):
        super().__init__()
        self.composer = dspy.ChainOfThought(
            "objective: str, constraints: str, format_instructions: str -> composed_prompt: str"
        )
    
    def forward(self, objective: str, constraints: str, format_instructions: str) -> dspy.Prediction:
        """Compose a final prompt from structured components.
        
        Args:
            objective: Task objective
            constraints: Task constraints
            format_instructions: Format requirements
            
        Returns:
            Prediction with composed prompt
        """
        return self.composer(
            objective=objective,
            constraints=constraints,
            format_instructions=format_instructions
        )


class MultiStagePromptXpert(dspy.Module):
    """Multi-stage prompt optimization with validation and refinement."""
    
    def __init__(self, max_refinement_iterations: int = 2):
        super().__init__()
        self.optimizer = PromptXpertProgram(use_validation=True)
        self.refiner = PromptRefiner()
        self.max_iterations = max_refinement_iterations
    
    def forward(self, initial_prompt: str) -> dspy.Prediction:
        """Multi-stage optimization with optional refinement.
        
        Args:
            initial_prompt: The original prompt to optimize
            
        Returns:
            Final optimized prediction after potential refinement
        """
        # Initial optimization
        result = self.optimizer(initial_prompt=initial_prompt)
        
        # Iterative refinement if validation suggests improvements
        iteration = 0
        while (iteration < self.max_iterations and 
               hasattr(result, 'validation_suggestions') and 
               result.validation_suggestions):
            
            refined = self.refiner(
                current_prompt=result.optimized_prompt,
                feedback=result.validation_suggestions
            )
            
            if hasattr(refined, 'refined_prompt'):
                result.optimized_prompt = refined.refined_prompt
                result.refinement_iteration = iteration + 1
                result.last_improvements = getattr(refined, 'improvements', '')
            
            iteration += 1
        
        return result
