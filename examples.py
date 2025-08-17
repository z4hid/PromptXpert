"""Example usage of PromptXpert for prompt optimization."""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def example_basic_usage():
    """Example of basic PromptXpert usage."""
    print("=== Basic PromptXpert Usage Example ===\n")
    
    # Note: This is a conceptual example. In practice, you'd need to:
    # 1. Set GROQ_API_KEY environment variable
    # 2. Install dependencies: pip install -r requirements.txt
    # 3. Run the actual training and evaluation
    
    print("1. Training a PromptXpert model:")
    print("   python -m src.train")
    print("   # This will:")
    print("   # - Load dataset from data/prompts_dataset.csv")
    print("   # - Split into train/val/test sets") 
    print("   # - Use MIPROv2 to optimize prompts")
    print("   # - Save model to outputs/compiled_promptxpert_model")
    print("   # - Track everything in MLflow")
    
    print("\n2. Evaluating the model:")
    print("   python -m src.eval")
    print("   # This will:")
    print("   # - Load the trained model")
    print("   # - Run comprehensive evaluation on test set")
    print("   # - Compare against baseline (identity transform)")
    print("   # - Generate detailed analysis report")
    
    print("\n3. Using the CLI for optimization:")
    print('   python -m src.cli optimize "Write a summary"')
    print("   # Output: Optimized prompt with structure and details")
    
    print("\n4. Batch optimization:")
    print("   python -m src.cli batch-optimize input.json output.json")
    print("   # Optimize multiple prompts from a file")


def example_configuration():
    """Example of configuration options."""
    print("\n=== Configuration Example ===\n")
    
    print("Environment variables (.env file):")
    print("GROQ_API_KEY=your_api_key_here")
    print("PROMPTXPERT_PRIMARY_MODEL=llama3-8b-8192")
    print("PROMPTXPERT_JUDGE_MODEL=llama3-70b-8192")
    print("MLFLOW_TRACKING_URI=http://localhost:5000")
    
    print("\nProgrammatic configuration:")
    print("""
from src.config import PromptXpertConfig

config = PromptXpertConfig()
config.model.primary_model = "llama3-8b-8192"
config.optimization.num_trials = 25
config.metrics.clarity_weight = 0.4  # Emphasize clarity
""")


def example_custom_dataset():
    """Example of creating a custom dataset."""
    print("\n=== Custom Dataset Example ===\n")
    
    print("Create data/prompts_dataset.csv with format:")
    print("""
initial_prompt,optimized_prompt
"Write a summary","Write a comprehensive summary that captures the main points, key arguments, and important conclusions. Structure the summary with clear paragraphs and use bullet points for lists."
"Debug this code","Systematically debug this code by: 1) Identifying the specific error or unexpected behavior, 2) Checking variable values and data types, 3) Testing edge cases, 4) Providing a corrected version with explanations."
""")
    
    print("The system will automatically:")
    print("- Validate the dataset format")
    print("- Split into train/validation/test sets") 
    print("- Augment if dataset is small (<20 examples)")
    print("- Shuffle data to prevent order bias")


def example_mlflow_usage():
    """Example of MLflow integration."""
    print("\n=== MLflow Integration Example ===\n")
    
    print("Start MLflow UI:")
    print("mlflow ui --port 5000")
    print("# Open http://localhost:5000 in browser")
    
    print("\nWhat gets tracked:")
    print("✓ All configuration parameters")
    print("✓ Dataset statistics and splits")
    print("✓ Optimization progress (trial scores)")
    print("✓ Model artifacts and versioning")
    print("✓ Evaluation metrics and comparisons")
    print("✓ Example predictions for analysis")
    print("✓ Environment information (Python version, etc.)")


def example_expected_output():
    """Example of expected system output."""
    print("\n=== Expected Output Example ===\n")
    
    print("Training Results:")
    print("Baseline score: 0.482")
    print("Optimized score: 0.763") 
    print("Improvement: 0.281 (58.3%)")
    print("Model saved to: outputs/compiled_promptxpert_model")
    
    print("\nOptimization Example:")
    print("Input: 'Write a summary'")
    print("Output:")
    print("  Objective: Create a comprehensive summary of the provided content")
    print("  Constraints:")
    print("  • Capture main points and key arguments")  
    print("  • Maintain logical flow and structure")
    print("  • Keep length appropriate to source material")
    print("  Format: Use clear paragraphs with topic sentences")
    print("  Optimized: 'Write a comprehensive summary of the provided content that captures the main points, key arguments, and important conclusions. Structure your summary with clear paragraphs, each starting with a topic sentence. Ensure the length is appropriate to the source material while maintaining logical flow throughout.'")


def example_troubleshooting():
    """Example troubleshooting guide."""
    print("\n=== Troubleshooting Guide ===\n")
    
    print("Common Issues:")
    print("1. 'GROQ_API_KEY not found'")
    print("   → Set environment variable: export GROQ_API_KEY=your_key")
    print("   → Or add to .env file")
    
    print("\n2. 'Dataset too small' warning")
    print("   → Add more examples to data/prompts_dataset.csv")
    print("   → System will auto-augment small datasets")
    
    print("\n3. 'Low improvement scores'")
    print("   → Increase optimization trials (--trials 30)")
    print("   → Use different models for primary/judge")
    print("   → Check dataset quality (diverse, realistic examples)")
    
    print("\n4. MLflow tracking issues")
    print("   → Check MLFLOW_TRACKING_URI environment variable")
    print("   → Ensure MLflow server is running")
    print("   → Verify write permissions in tracking directory")


def main():
    """Main example function."""
    print("🚀 PromptXpert Usage Examples and Guide 🚀")
    print("=" * 60)
    
    example_basic_usage()
    example_configuration()
    example_custom_dataset()
    example_mlflow_usage()
    example_expected_output()
    example_troubleshooting()
    
    print("\n" + "=" * 60)
    print("📝 Next Steps:")
    print("1. Set up your GROQ_API_KEY in .env file")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Run training: python -m src.train")
    print("4. Start optimizing prompts!")
    print("\n📚 For more details, see README.md")
    print("🐛 For issues, check the troubleshooting section above")


if __name__ == "__main__":
    main()