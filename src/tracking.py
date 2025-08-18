"""Centralized MLflow / DagsHub tracking setup utilities.

Environment variables (optional):
    DAGSHUB_REPO_OWNER      # e.g. 'z4hid'
    DAGSHUB_REPO_NAME       # e.g. 'PromptXpert'
    DAGSHUB_TOKEN           # personal access token with repo permissions
    MLFLOW_EXPERIMENT_NAME  # override default experiment name
    MLFLOW_TRACKING_URI     # standard MLflow URI fallback if not using DagsHub

If the three DagsHub vars are present we build the tracking URI
https://dagshub.com/{owner}/{repo}.mlflow and configure basic auth via
MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD.
"""
from __future__ import annotations

import logging
import os
import subprocess
from typing import Dict, Any

import mlflow

logger = logging.getLogger("prompt_xpert")


def _git(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def init_tracking() -> str:
    """Configure MLflow to use DagsHub if token + repo env vars provided.

    Returns the experiment name ultimately selected (for logging statements).
    Safe to call multiple times (idempotent).
    """
    owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo = os.getenv("DAGSHUB_REPO_NAME")
    token = os.getenv("DAGSHUB_TOKEN")

    if owner and repo and token:
        uri = f"https://dagshub.com/{owner}/{repo}.mlflow"
        # Basic auth expected by DagsHub's MLflow proxy
        os.environ.setdefault("MLFLOW_TRACKING_USERNAME", owner)
        os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", token)
        try:
            mlflow.set_tracking_uri(uri)
            logger.info(f"Using DagsHub MLflow tracking at {uri}")
        except Exception as e:
            logger.warning(f"Failed to set DagsHub tracking URI: {e}")
    else:
        if os.getenv("MLFLOW_TRACKING_URI"):
            logger.info(f"Using MLflow tracking at {os.getenv('MLFLOW_TRACKING_URI')}")
        else:
            logger.info("No remote tracking configured; MLflow will use local ./mlruns store")

    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "DSPy-Optimization")
    try:
        mlflow.set_experiment(exp_name)
        logger.info(f"MLflow experiment set to '{exp_name}'")
    except Exception as e:
        logger.warning(f"Could not set MLflow experiment '{exp_name}': {e}")
    return exp_name


def base_tags(cfg) -> Dict[str, Any]:
    """Collect reproducibility tags for the parent pipeline run.

    Includes git commit, dirty flag, dataset path, and optional run context.
    """
    commit = _git(["git", "rev-parse", "HEAD"]) or "unknown"
    dirty = bool(_git(["git", "status", "--porcelain"]))
    tags: Dict[str, Any] = {
        "git_commit": commit,
        "git_dirty": str(dirty),
        "dataset_csv": getattr(cfg, "dataset_csv", "unknown"),
        "artifacts_dir": getattr(cfg, "artifacts_dir", "artifacts"),
        "module": "prompt_xpert_pipeline",
    }
    # Copy selected config hyperparams (avoid full dictionary explosion as tags)
    for k in ["main_model", "judge_model", "auto_level", "minibatch_size", "seed"]:
        if hasattr(cfg, k):
            tags[f"cfg_{k}"] = getattr(cfg, k)
    return tags


__all__ = ["init_tracking", "base_tags"]
