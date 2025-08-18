import json
import os
import time
import shutil
import mlflow
import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import MIPROv2
from .config import config
from .logging_setup import logger, configure_logging_and_mlflow
from .lm_init import init_lms
from .data_utils import ensure_dataset, load_examples, split_train_dev
from .program import PromptXpertProgram
from .metrics import MultiCriteriaPromptMetric
from .tracking import init_tracking, base_tags


def run_pipeline():
    configure_logging_and_mlflow()
    # Parent run to unify nested compilation + evaluation runs
    parent_run = None
    try:
        init_tracking()
        parent_run = mlflow.start_run(run_name="pipeline", nested=False)
        mlflow.set_tags(base_tags(config))
    except Exception as e:
        logger.warning(f"Failed to start parent MLflow run: {e}")

    print("Step 1: Initial DSPy program and signature defined.")

    ensure_dataset(config.dataset_csv)
    examples = load_examples(config.dataset_csv)
    trainset, devset = split_train_dev(examples)
    print(f"Step 2: Loaded {len(trainset)} training examples and {len(devset)} development examples.")
    if trainset:
        print("Example trainset entries:")
        for i, example in enumerate(trainset[:min(len(trainset), 2)]):
            print(f"  Entry {i+1} (initial_prompt): {example.initial_prompt}")
            print(f"  Entry {i+1} (optimized_prompt): {example.optimized_prompt}")
    if devset:
        print("Example devset entries:")
        for i, example in enumerate(devset[:min(len(devset), 2)]):
            print(f"  Entry {i+1} (initial_prompt): {example.initial_prompt}")
            print(f"  Entry {i+1} (optimized_prompt): {example.optimized_prompt}")

    # (Experiment handled in tracking init; retain graceful fallback)

    lm, judge_lm = init_lms()
    metric = MultiCriteriaPromptMetric(judge_lm)
    print("Step 3: LLM judge metric defined.")

    student = PromptXpertProgram()

    print("\nStep 4: Starting MIPROv2 compilation...")
    with mlflow.start_run(run_name="MIPROv2 Compilation", nested=True):
        mlflow.log_params(vars(config))
        teleprompter = MIPROv2(
            metric=metric,
            prompt_model=lm,
            task_model=lm,
            max_bootstrapped_demos=config.max_bootstrapped_demos,
            max_labeled_demos=config.max_labeled_demos,
            auto=config.auto_level,
            num_threads=config.num_threads,
            verbose=True,
            seed=config.seed,
        )
        try:
            compiled_program = teleprompter.compile(
                student=student,
                trainset=trainset,
                valset=devset,
                minibatch_size=config.minibatch_size,
                requires_permission_to_run=False,
            )
            print("MIPROv2 compilation completed.")
            mlflow.log_param("compilation_status", "Success")
        except Exception as e:
            logger.error(f"MIPROv2 compilation failed: {e}")
            mlflow.log_param("compilation_status", "Failed")
            mlflow.log_param("compilation_error", str(e))
            print("Compilation failed. Check logs for details.")
            return None, None

    print("\nEvaluating the optimized program...")
    with mlflow.start_run(run_name="Optimized Program Evaluation", nested=True):
        evaluator = Evaluate(
            devset=devset,
            metric=metric,
            num_threads=config.num_threads,
            display_progress=True,
            display_table=True,
        )
        avg_score = evaluator(compiled_program)
        print("Optimized program evaluation completed.")
        mlflow.log_metric("average_metric_score", avg_score if avg_score is not None else 0.0)
        mlflow.log_param("evaluation_dataset_size", len(devset))

    # ----- Artifact versioning -----
    config.ensure_artifacts_dir()
    raw_score = avg_score if avg_score is not None else 0.0
    try:
        numeric_score = float(getattr(raw_score, 'value', raw_score))
    except Exception:
        numeric_score = 0.0
    ts = time.strftime('%Y%m%d-%H%M%S')
    version_basename = f"promptxpert_{ts}_score{numeric_score:.4f}".replace('.', '-')
    # State-only save (JSON) if enabled
    program_path = os.path.join(config.artifacts_dir, version_basename + '.json')
    state_saved = False
    if config.save_state_only:
        compiled_program.save(program_path, save_program=False)
        state_saved = True
        print(f"\nStep 5: State-only program state saved to {program_path}")
    # Whole program save (directory) if enabled
    whole_dir = os.path.join(config.artifacts_dir, version_basename + '_dir')
    whole_saved_path = None
    if config.save_whole_program:
        compiled_program.save(whole_dir + '/', save_program=True, modules_to_serialize=[])
        whole_saved_path = whole_dir
        print(f"Whole program (architecture+state) saved to {whole_dir}/")

    metadata = {
        "avg_score": numeric_score,
        "trainset_size": len(trainset),
        "devset_size": len(devset),
        "config": vars(config),
        "program_file": os.path.basename(program_path) if state_saved else None,
        "whole_program_dir": whole_saved_path,
        "timestamp": ts,
    }
    meta_path = os.path.join(config.artifacts_dir, version_basename + '_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {meta_path}")

    # Update legacy single-file copies for backward compatibility
    if state_saved:
        compiled_program.save(config.save_path, save_program=False)
    with open(config.metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Maintain a 'best' copy: pick highest score among artifacts
    best_link = os.path.join(config.artifacts_dir, 'best_program.json')
    best_meta = os.path.join(config.artifacts_dir, 'best_program_meta.json')
    # Determine current best
    try:
        current_best_score = -1.0
        if os.path.exists(best_meta):
            with open(best_meta) as bf:
                bm = json.load(bf)
                current_best_score = float(bm.get('avg_score', -1.0))
        if numeric_score >= current_best_score:
            # Prefer state-only program for best link; if not saved, point to whole program dir marker file
            if state_saved:
                shutil.copy2(program_path, best_link)
            else:
                # create a lightweight pointer file
                with open(best_link, 'w') as bf:
                    json.dump({"redirect_to_whole_program_dir": whole_saved_path, "avg_score": numeric_score}, bf)
            shutil.copy2(meta_path, best_meta)
            print(f"Updated best program to {program_path if state_saved else whole_saved_path}")
    except Exception as e:
        logger.warning(f"Could not update best program: {e}")

    loaded_program = PromptXpertProgram()
    # Load from state JSON if we saved it, else from whole program directory using dspy.load
    if state_saved:
        loaded_program.load(program_path)
        print(f"Program loaded from state file {program_path}")
    elif whole_saved_path:
        # Whole-program load returns a program object; reassign
        try:
            loaded_program = dspy.load(whole_saved_path + '/')
            print(f"Whole program loaded from {whole_saved_path}/")
        except Exception as e:
            logger.error(f"Failed to load whole program: {e}")

    # Log artifacts (in parent run if active)
    try:
        if mlflow.active_run() and parent_run:
            # Need to ensure we are in the parent run context
            if mlflow.active_run().info.run_id != parent_run.info.run_id:
                mlflow.end_run()  # end evaluation nested run
                mlflow.start_run(run_id=parent_run.info.run_id)
            # Log created artifacts
            if state_saved and os.path.exists(program_path):
                mlflow.log_artifact(program_path, artifact_path="program")
            if os.path.exists(meta_path):
                mlflow.log_artifact(meta_path, artifact_path="program")
            best_meta_path = os.path.join(config.artifacts_dir, 'best_program_meta.json')
            if os.path.exists(best_meta_path):
                mlflow.log_artifact(best_meta_path, artifact_path="best")
            best_prog_path = os.path.join(config.artifacts_dir, 'best_program.json')
            if os.path.exists(best_prog_path):
                mlflow.log_artifact(best_prog_path, artifact_path="best")
            mlflow.log_metric("final_average_metric_score", numeric_score)
    except Exception as e:
        logger.warning(f"Failed to log artifacts to MLflow: {e}")
    finally:
        if parent_run:
            try:
                mlflow.end_run()
            except Exception:
                pass

    print("\nStep 6: Running inference on new prompts.")
    return compiled_program, loaded_program
