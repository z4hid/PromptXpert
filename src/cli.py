"""Command-line interface for PromptXpert."""

import typer
import json
import os
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .config import load_config_from_env
from .train import train_prompt_optimizer
from .eval import run_evaluation, evaluate_model_on_new_prompts, load_trained_model

app = typer.Typer(help="PromptXpert: Advanced prompt optimization toolkit using DSPy")
console = Console()


@app.command()
def train(
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to configuration file"),
    primary_model: Optional[str] = typer.Option(None, "--primary-model", help="Primary model name"),
    judge_model: Optional[str] = typer.Option(None, "--judge-model", help="Judge model name"),
    num_trials: Optional[int] = typer.Option(None, "--trials", help="Number of optimization trials"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
):
    """Train a PromptXpert model for prompt optimization."""
    try:
        # Load configuration
        config = load_config_from_env()
        
        # Override with CLI arguments
        if primary_model:
            config.model.primary_model = primary_model
        if judge_model:
            config.model.judge_model = judge_model
        if num_trials:
            config.optimization.num_trials = num_trials
        if output_dir:
            config.output_dir = output_dir
        
        console.print(Panel.fit("Starting PromptXpert Training", style="bold green"))
        
        # Check for API key
        if not os.getenv(config.model.api_key_env):
            console.print(f"[red]Error: {config.model.api_key_env} environment variable not set[/red]")
            raise typer.Exit(1)
        
        # Display configuration
        config_table = Table(title="Training Configuration")
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="magenta")
        
        config_table.add_row("Primary Model", config.model.primary_model)
        config_table.add_row("Judge Model", config.model.judge_model)
        config_table.add_row("Optimization Mode", config.optimization.auto_mode)
        config_table.add_row("Number of Trials", str(config.optimization.num_trials))
        config_table.add_row("Output Directory", config.output_dir)
        config_table.add_row("Dataset", config.data.dataset_path)
        
        console.print(config_table)
        
        # Run training
        results = train_prompt_optimizer(config)
        
        # Display results
        results_table = Table(title="Training Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")
        
        results_table.add_row("Model Path", results['model_path'])
        results_table.add_row("Baseline Score", f"{results['baseline_score']:.3f}")
        results_table.add_row("Optimized Score", f"{results['optimized_score']:.3f}")
        results_table.add_row("Improvement", f"{results['improvement']:.3f}")
        results_table.add_row("Relative Improvement", f"{results['relative_improvement_percent']:.1f}%")
        results_table.add_row("Training Examples", str(results['train_size']))
        results_table.add_row("Validation Examples", str(results['val_size']))
        results_table.add_row("Test Examples", str(results['test_size']))
        
        console.print(results_table)
        console.print(Panel.fit("Training completed successfully!", style="bold green"))
        
    except Exception as e:
        console.print(f"[red]Training failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def evaluate(
    model_path: Optional[str] = typer.Option(None, "--model", "-m", help="Path to trained model"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
):
    """Evaluate a trained PromptXpert model."""
    try:
        # Load configuration
        config = load_config_from_env()
        
        if output_dir:
            config.output_dir = output_dir
        
        console.print(Panel.fit("Starting PromptXpert Evaluation", style="bold blue"))
        
        # Check for API key
        if not os.getenv(config.model.api_key_env):
            console.print(f"[red]Error: {config.model.api_key_env} environment variable not set[/red]")
            raise typer.Exit(1)
        
        # Run evaluation
        results = run_evaluation(config, model_path)
        
        # Display results
        results_table = Table(title="Evaluation Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Score", style="green")
        
        results_table.add_row("Multi-Criteria Score", f"{results.get('multi_criteria_score', 0):.3f}")
        results_table.add_row("Legacy Score", f"{results.get('legacy_score', 0):.3f}")
        results_table.add_row("Heuristic Score", f"{results.get('heuristic_score', 0):.3f}")
        results_table.add_row("Composite Score", f"{results.get('composite_score', 0):.3f}")
        
        if 'baseline_comparison' in results:
            comp = results['baseline_comparison']
            results_table.add_row("", "")  # Separator
            results_table.add_row("Baseline Score", f"{comp.get('baseline_score', 0):.3f}")
            results_table.add_row("Optimized Score", f"{comp.get('optimized_score', 0):.3f}")
            results_table.add_row("Improvement", f"{comp.get('absolute_improvement', 0):.3f}")
            results_table.add_row("Relative Improvement", f"{comp.get('relative_improvement_percent', 0):.1f}%")
        
        console.print(results_table)
        console.print(f"[green]Detailed report saved to: {config.output_dir}/evaluation_report.json[/green]")
        console.print(Panel.fit("Evaluation completed successfully!", style="bold blue"))
        
    except Exception as e:
        console.print(f"[red]Evaluation failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def optimize(
    prompt: str = typer.Argument(..., help="Prompt to optimize"),
    model_path: Optional[str] = typer.Option(None, "--model", "-m", help="Path to trained model"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Output file for results"),
):
    """Optimize a single prompt using a trained model."""
    try:
        # Load configuration
        config = load_config_from_env()
        
        # Check for API key
        if not os.getenv(config.model.api_key_env):
            console.print(f"[red]Error: {config.model.api_key_env} environment variable not set[/red]")
            raise typer.Exit(1)
        
        # Load model
        if model_path is None:
            model_path = Path(config.output_dir) / "compiled_promptxpert_model"
        
        console.print(Panel.fit("Optimizing Prompt", style="bold cyan"))
        console.print(f"[cyan]Input prompt:[/cyan] {prompt}")
        
        model = load_trained_model(str(model_path))
        
        # Setup language models for inference
        from .train import setup_language_models
        primary_lm, _ = setup_language_models(config)
        
        # Optimize prompt
        results = evaluate_model_on_new_prompts(model, [prompt])
        
        if results and 'error' not in results[0]:
            result = results[0]
            
            # Display results
            console.print(f"[green]Optimized prompt:[/green] {result['optimized_prompt']}")
            
            if result.get('objective'):
                console.print(f"[yellow]Objective:[/yellow] {result['objective']}")
            if result.get('constraints'):
                console.print(f"[yellow]Constraints:[/yellow] {result['constraints']}")
            if result.get('format_instructions'):
                console.print(f"[yellow]Format Instructions:[/yellow] {result['format_instructions']}")
            
            # Save to file if requested
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                console.print(f"[green]Results saved to: {output_file}[/green]")
            
            console.print(Panel.fit("Optimization completed successfully!", style="bold green"))
        else:
            error_msg = results[0].get('error', 'Unknown error') if results else 'No results'
            console.print(f"[red]Optimization failed: {error_msg}[/red]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]Optimization failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def batch_optimize(
    input_file: str = typer.Argument(..., help="Input file with prompts (JSON or text)"),
    output_file: str = typer.Argument(..., help="Output file for results"),
    model_path: Optional[str] = typer.Option(None, "--model", "-m", help="Path to trained model"),
):
    """Optimize multiple prompts from a file."""
    try:
        # Load configuration
        config = load_config_from_env()
        
        # Check for API key
        if not os.getenv(config.model.api_key_env):
            console.print(f"[red]Error: {config.model.api_key_env} environment variable not set[/red]")
            raise typer.Exit(1)
        
        # Load prompts
        input_path = Path(input_file)
        if not input_path.exists():
            console.print(f"[red]Input file not found: {input_file}[/red]")
            raise typer.Exit(1)
        
        if input_path.suffix.lower() == '.json':
            with open(input_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    prompts = data
                elif isinstance(data, dict) and 'prompts' in data:
                    prompts = data['prompts']
                else:
                    prompts = [str(data)]
        else:
            with open(input_file, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]
        
        console.print(f"[cyan]Processing {len(prompts)} prompts...[/cyan]")
        
        # Load model
        if model_path is None:
            model_path = Path(config.output_dir) / "compiled_promptxpert_model"
        
        model = load_trained_model(str(model_path))
        
        # Setup language models
        from .train import setup_language_models
        primary_lm, _ = setup_language_models(config)
        
        # Optimize prompts
        results = evaluate_model_on_new_prompts(model, prompts)
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Display summary
        successful = len([r for r in results if 'error' not in r])
        failed = len(results) - successful
        
        summary_table = Table(title="Batch Optimization Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Count", style="green")
        
        summary_table.add_row("Total Prompts", str(len(prompts)))
        summary_table.add_row("Successful", str(successful))
        summary_table.add_row("Failed", str(failed))
        
        console.print(summary_table)
        console.print(f"[green]Results saved to: {output_file}[/green]")
        console.print(Panel.fit("Batch optimization completed!", style="bold green"))
        
    except Exception as e:
        console.print(f"[red]Batch optimization failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def info():
    """Display PromptXpert information and configuration."""
    try:
        config = load_config_from_env()
        
        # Display general info
        console.print(Panel.fit("PromptXpert Information", style="bold magenta"))
        
        # Configuration table
        config_table = Table(title="Current Configuration")
        config_table.add_column("Category", style="cyan")
        config_table.add_column("Parameter", style="yellow")
        config_table.add_column("Value", style="green")
        
        # Model config
        config_table.add_row("Model", "Primary Model", config.model.primary_model)
        config_table.add_row("", "Judge Model", config.model.judge_model)
        config_table.add_row("", "Primary Temperature", str(config.model.primary_temperature))
        config_table.add_row("", "Judge Temperature", str(config.model.judge_temperature))
        
        # Optimization config
        config_table.add_row("Optimization", "Max Labeled Demos", str(config.optimization.max_labeled_demos))
        config_table.add_row("", "Max Bootstrapped Demos", str(config.optimization.max_bootstrapped_demos))
        config_table.add_row("", "Auto Mode", config.optimization.auto_mode)
        config_table.add_row("", "Num Trials", str(config.optimization.num_trials))
        
        # Data config
        config_table.add_row("Data", "Dataset Path", config.data.dataset_path)
        config_table.add_row("", "Train Ratio", str(config.data.train_ratio))
        config_table.add_row("", "Val Ratio", str(config.data.val_ratio))
        config_table.add_row("", "Test Ratio", str(config.data.test_ratio))
        
        # Global config
        config_table.add_row("Global", "Output Directory", config.output_dir)
        config_table.add_row("", "Random Seed", str(config.random_seed))
        config_table.add_row("", "Cache Enabled", str(config.cache_enabled))
        
        console.print(config_table)
        
        # Check API key
        api_key_status = "✓ Set" if os.getenv(config.model.api_key_env) else "✗ Not Set"
        console.print(f"[cyan]API Key ({config.model.api_key_env}):[/cyan] {api_key_status}")
        
        # Check if model exists
        model_path = Path(config.output_dir) / "compiled_promptxpert_model"
        model_status = "✓ Found" if model_path.exists() else "✗ Not Found"
        console.print(f"[cyan]Trained Model:[/cyan] {model_status}")
        
        if model_path.exists():
            console.print(f"[green]Model path: {model_path}[/green]")
        
    except Exception as e:
        console.print(f"[red]Failed to load configuration: {e}[/red]")
        raise typer.Exit(1)


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()