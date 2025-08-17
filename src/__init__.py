"""PromptXpert: Advanced prompt optimization toolkit using DSPy."""

__version__ = "1.0.0"
__author__ = "Md. Zahid Hasan"

from .config import PromptXpertConfig, load_config_from_env
from .module import PromptXpertProgram, BaselineProgram, MultiStagePromptXpert
from .metrics import PromptMetrics, create_metric_function
from .utils import prepare_dataset, set_reproducibility_seeds
from .train import train_prompt_optimizer
from .eval import run_evaluation, evaluate_model_on_new_prompts

__all__ = [
    "PromptXpertConfig",
    "load_config_from_env", 
    "PromptXpertProgram",
    "BaselineProgram",
    "MultiStagePromptXpert",
    "PromptMetrics",
    "create_metric_function",
    "prepare_dataset",
    "set_reproducibility_seeds",
    "train_prompt_optimizer",
    "run_evaluation",
    "evaluate_model_on_new_prompts"
]