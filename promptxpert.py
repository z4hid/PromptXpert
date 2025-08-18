"""Entry point script that wires modular DSPy PromptXpert pipeline."""

import argparse
from src.pipeline import run_pipeline
from src.program import PromptXpertProgram
from src.inference import load_program, optimize_prompt
from src.config import config


def main():
    parser = argparse.ArgumentParser(description="PromptXpert DSPy Optimization")
    parser.add_argument("--mode", choices=["train", "infer"], default="train", help="Run full training pipeline or just inference")
    parser.add_argument("--prompt", type=str, help="Prompt to optimize in inference mode", default=None)
    parser.add_argument("--model_path", type=str, help="Explicit path to a saved program JSON inside artifacts", default=None)
    parser.add_argument("--no_best", action="store_true", help="Do not auto-select best model; use legacy or explicit path")
    parser.add_argument("--dataset", type=str, help="Path to dataset CSV (default data/prompts_dataset.csv)", default=None)
    args = parser.parse_args()

    if args.mode == "train":
        if args.dataset:
            config.set_dataset(args.dataset)
        compiled_program, loaded_program = run_pipeline()
        if loaded_program:
            sample_prompt = args.prompt or "Write a social media post for a new fitness app called Bosho."
            optimized = optimize_prompt(sample_prompt, loaded_program)
            print(f"Original Prompt: {sample_prompt}")
            print(f"Optimized Prompt: {optimized}")
    else:  # infer
        if not args.prompt:
            raise SystemExit("--prompt is required in infer mode")
        program = load_program(path=args.model_path, best=not args.no_best)
        optimized = optimize_prompt(args.prompt, program)
        print(optimized)


if __name__ == "__main__":
    main()