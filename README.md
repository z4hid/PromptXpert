# PromptXpert

PromptXpert is an advanced automated prompt-engineering toolkit built on top of DSPy. It trains a sophisticated DSPy program that takes any initial prompt and returns an optimized version judged to be clearer, more complete, and more effective for LLM use.

## Features

- **Multi-criteria optimization**: Uses weighted combination of clarity, specificity, brevity, and safety metrics
- **Baseline comparison**: Automatically compares optimized prompts against baseline (identity transformation)
- **MLflow integration**: Full experiment tracking and model management
- **Reproducible results**: Comprehensive seed management and environment logging
- **Multiple evaluation metrics**: LLM judge, heuristic, and composite scoring
- **Structured output**: Generates objective, constraints, format instructions, and final optimized prompt
- **CLI interface**: Easy-to-use command-line tools for training, evaluation, and inference
- **Comprehensive dataset handling**: Automatic splitting, shuffling, and augmentation

## Architecture

PromptXpert implements the complete DSPy workflow:

1. **Signature Definition**: Structured I/O with `PromptOptimization` and `MultiCriteriaAssessment` signatures
2. **Module Implementation**: `PromptXpertProgram` using Chain of Thought reasoning
3. **Dataset Preparation**: CSV-based dataset with train/validation/test splits
4. **Multi-criteria Metrics**: LLM judge with weighted scoring across multiple dimensions
5. **Optimization**: MIPROv2 teleprompting for instruction and demonstration optimization
6. **Evaluation**: Comprehensive evaluation with baseline comparison
7. **Persistence**: Model saving/loading with MLflow artifact management
8. **Inference**: Production-ready prompt optimization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/z4hid/PromptXpert.git
cd PromptXpert
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.template .env
# Edit .env with your Groq API key
```

## Quick Start

### 1. Training

Train a PromptXpert model:

```bash
# Using Python module
python -m src.train

# Using CLI
python -m src.cli train --trials 25 --output ./models
```

### 2. Evaluation

Evaluate a trained model:

```bash
# Using Python module
python -m src.eval

# Using CLI
python -m src.cli evaluate --model ./models/compiled_promptxpert_model
```

### 3. Optimize Prompts

Optimize individual prompts:

```bash
# Single prompt
python -m src.cli optimize "Write a summary" --model ./models/compiled_promptxpert_model

# Batch optimization
python -m src.cli batch-optimize prompts.json results.json
```

## Configuration

PromptXpert uses a comprehensive configuration system. Key settings:

```python
from src.config import PromptXpertConfig

config = PromptXpertConfig()
config.model.primary_model = "llama3-8b-8192"  # Main model
config.model.judge_model = "llama3-70b-8192"   # Judge model (different for bias reduction)
config.optimization.num_trials = 20            # MIPROv2 trials
config.optimization.auto_mode = "medium"       # light, medium, heavy
config.metrics.clarity_weight = 0.35           # Multi-criteria weights
```

## MLflow Integration

PromptXpert provides comprehensive experiment tracking:

- **Configuration logging**: All hyperparameters and settings
- **Dataset tracking**: Size, splits, and quality metrics
- **Optimization progress**: Real-time trial scores and improvements
- **Model artifacts**: Trained models with versioning
- **Evaluation results**: Multi-metric assessment and comparisons
- **Example predictions**: Sample inputs/outputs for analysis

Start MLflow UI:
```bash
mlflow ui --port 5000
```

## Dataset Format

Use CSV format with `initial_prompt` and `optimized_prompt` columns:

```csv
initial_prompt,optimized_prompt
"Write a summary","Write a comprehensive summary highlighting main points..."
"Explain AI","Provide a clear explanation of artificial intelligence..."
```

## Advanced Usage

### Multi-stage Optimization

```python
from src.module import MultiStagePromptXpert

# Use multi-stage optimization with validation and refinement
multi_stage_model = MultiStagePromptXpert(max_refinement_iterations=2)
result = multi_stage_model(initial_prompt="Your prompt here")
```

### Custom Metrics

```python
from src.metrics import PromptMetrics

# Use different metric types
metrics = PromptMetrics(judge_lm, config.metrics)
score = metrics.composite_metric(example, prediction)  # Combines multiple approaches
```

### Programmatic Usage

```python
from src import train_prompt_optimizer, run_evaluation, load_config_from_env

# Train model
config = load_config_from_env()
results = train_prompt_optimizer(config)

# Evaluate model
eval_results = run_evaluation(config, model_path="path/to/model")

# Optimize new prompts
from src.eval import evaluate_model_on_new_prompts, load_trained_model
model = load_trained_model("path/to/model")
optimized = evaluate_model_on_new_prompts(model, ["Your prompt"])
```

## Performance Tips

1. **Model Selection**: Use different model families for primary and judge models to reduce bias
2. **Dataset Size**: Minimum 20-30 examples recommended; augmentation available for smaller datasets
3. **Optimization Trials**: Start with 15-20 trials, increase for better results
4. **Caching**: Enable caching to avoid duplicate API calls during experimentation
5. **Batch Size**: Adjust minibatch_size based on your API rate limits

## Evaluation Metrics

PromptXpert provides multiple evaluation approaches:

- **Multi-criteria LLM Judge**: Weighted combination of clarity, specificity, brevity, and safety
- **Legacy Single Score**: Overall improvement rating (0.0-1.0)
- **Heuristic Analysis**: Rule-based scoring for sanity checking
- **Composite Score**: Combination of LLM and heuristic approaches

## API Integration

PromptXpert currently supports:
- **Groq**: Fast inference with Llama models
- **OpenAI**: GPT models (configure in model settings)

Add support for other providers by extending the language model configuration.

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use PromptXpert in your research, please cite:

```bibtex
@misc{promptxpert2025,
  title={PromptXpert: Advanced Prompt Optimization with DSPy},
  author={Md. Zahid Hasan},
  year={2025},
  url={https://github.com/z4hid/PromptXpert}
}
```

## Acknowledgments

- Built on the excellent [DSPy framework](https://github.com/stanfordnlp/dspy)
- Inspired by research in automated prompt engineering and teleprompting
- Uses MLflow for comprehensive experiment tracking
