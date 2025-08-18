import dspy

class PromptOptimization(dspy.Signature):
    initial_prompt: str = dspy.InputField(desc="The initial prompt provided by the user.")
    optimized_prompt: str = dspy.OutputField(desc="A clearer, more specific, more detailed and more actionable version of the prompt.")

class PromptXpertProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.optimizer = dspy.ChainOfThought(PromptOptimization)

    def forward(self, initial_prompt: str):
        pred = self.optimizer(initial_prompt=initial_prompt)
        return dspy.Prediction(optimized_prompt=pred.optimized_prompt)
