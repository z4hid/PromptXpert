"""FastAPI + Telegram Bot interface for PromptXpert.

Features:
- Loads the current *best* compiled program artifact (see artifacts/best_program.json or fallback).
- Provides a REST API to optimize prompts.
- Exposes a Telegram bot (python-telegram-bot) that replies with optimized prompts.

Environment Variables:
- GEMINI_API_KEY: API key required by underlying LM initialization (see src/lm_init.py)
- TELEGRAM_BOT_TOKEN: (optional if you don't need Telegram) token for the bot.
- PROGRAM_PATH: (optional) explicit path to a program artifact (file or directory). If unset, best artifact auto-detected.
- RELOAD_INTERVAL_SECONDS: (optional int) if set >0, background task will watch for newer best artifact and hot-reload.

Run (API only):
    uvicorn app:app --host 0.0.0.0 --port 8000

Run (API + Telegram): ensure TELEGRAM_BOT_TOKEN is set before starting.

NOTE: The telegram bot processes each incoming message as an initial_prompt and replies ONLY with the optimized prompt text.
"""
from __future__ import annotations

import os
import sys
import time
import threading
import logging
from pathlib import Path
from typing import Optional
import csv
from datetime import datetime, timezone

import mlflow

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure local src/ is importable when running from repo root
CURRENT_DIR = Path(__file__).parent.resolve()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(CURRENT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR / "src"))

from src.inference import load_program, optimize_prompt  # noqa: E402
from src.config import config  # noqa: E402

# Telegram imports are optional; degrade gracefully if not installed.
try:
    from telegram import Update
    from telegram.ext import (Application, CommandHandler, MessageHandler,
                              ContextTypes, filters)
except Exception:  # pragma: no cover - optional dependency
    Update = None  # type: ignore
    Application = None  # type: ignore

logger = logging.getLogger("promptxpert.app")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)

app = FastAPI(title="PromptXpert API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Global State --------
_program = None
_program_path_used: Optional[str] = None
_program_mtime: float = 0.0
_program_lock = threading.RLock()
_stop_reload = threading.Event()
_reload_thread: Optional[threading.Thread] = None

_telegram_app: Optional[Application] = None  # Will be initialized on startup if token present

# Inference logging globals
_parent_run_id: Optional[str] = None
_inference_log_path: Optional[Path] = None  # detailed CSV (per inference, wide)
_inference_jsonl_path: Optional[Path] = None  # aggregated dataset JSONL (training style)
_inference_pairs_csv_path: Optional[Path] = None  # lightweight pairs CSV (initial,optimized)
_log_lock = threading.RLock()

def _init_inference_logging():
    """Initialize MLflow parent run and local flat-files for capturing inference pairs."""
    global _parent_run_id, _inference_log_path, _inference_jsonl_path, _inference_pairs_csv_path
    # Ensure tracking configured (DagsHub / MLflow)
    try:
        from src.tracking import init_tracking  # local import to avoid cycles
        init_tracking()
    except Exception as e:  # pragma: no cover
        logger.warning(f"init_tracking failed: {e}")
    # Start / resume a parent run
    disable_mlflow = os.getenv("DISABLE_MLFLOW_INFERENCE") == '1'
    if not disable_mlflow:
        try:
            active = mlflow.active_run()
            if active is None:
                parent = mlflow.start_run(run_name="inference_server", nested=False)
                _parent_run_id = parent.info.run_id
            else:
                _parent_run_id = active.info.run_id
        except Exception as e:  # pragma: no cover
            logger.warning(f"Could not start MLflow parent run: {e}")
    # Prepare CSV log file
    log_file = os.getenv("INFERENCE_LOG_FILE", str(Path(config.artifacts_dir) / "inference_log.csv"))
    _inference_log_path = Path(log_file)
    _inference_log_path.parent.mkdir(parents=True, exist_ok=True)
    if not _inference_log_path.exists():
        try:
            with _inference_log_path.open('w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp_utc","channel","initial_prompt","optimized_prompt","artifact_path"])
        except Exception as e:  # pragma: no cover
            logger.warning(f"Failed to create inference log file: {e}")

    # JSONL dataset file (each line: {ts, channel, initial_prompt, optimized_prompt, artifact})
    jsonl_file = os.getenv("INFERENCE_JSONL_FILE", str(Path(config.artifacts_dir) / "inference_dataset.jsonl"))
    _inference_jsonl_path = Path(jsonl_file)
    _inference_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    if not _inference_jsonl_path.exists():
        try:
            _inference_jsonl_path.touch()
        except Exception as e:  # pragma: no cover
            logger.warning(f"Failed to create JSONL dataset file: {e}")
    # Simple pairs CSV (initial_prompt,optimized_prompt) for quick download / preview
    pairs_csv_file = os.getenv("INFERENCE_PAIRS_CSV_FILE", str(Path(config.artifacts_dir) / "inference_pairs.csv"))
    _inference_pairs_csv_path = Path(pairs_csv_file)
    _inference_pairs_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not _inference_pairs_csv_path.exists():
        try:
            with _inference_pairs_csv_path.open('w', newline='', encoding='utf-8') as pf:
                writer = csv.writer(pf)
                writer.writerow(["initial_prompt","optimized_prompt"])
        except Exception as e:  # pragma: no cover
            logger.warning(f"Failed to create pairs CSV file: {e}")

def _log_inference_pair(initial_prompt: str, optimized_prompt: str, channel: str):
    """Record inference (initial, optimized) to CSV and MLflow nested run.

    channel: 'api' | 'telegram'
    """
    ts = datetime.now().replace(tzinfo=timezone.utc).isoformat()
    ts_safe = ts.replace(':', '-').replace('T', '_')
    artifact = _program_path_used
    # Append to CSV + JSONL under lock
    try:
        with _log_lock:
            if _inference_log_path is not None:
                with _inference_log_path.open('a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([ts, channel, initial_prompt, optimized_prompt, artifact])
            if _inference_jsonl_path is not None:
                import json
                with _inference_jsonl_path.open('a', encoding='utf-8') as jf:
                    json.dump({
                        "timestamp_utc": ts,
                        "channel": channel,
                        "initial_prompt": initial_prompt,
                        "optimized_prompt": optimized_prompt,
                        "artifact_path": artifact,
                    }, jf, ensure_ascii=False)
                    jf.write('\n')
            if _inference_pairs_csv_path is not None:
                with _inference_pairs_csv_path.open('a', newline='', encoding='utf-8') as pf:
                    writer = csv.writer(pf)
                    writer.writerow([initial_prompt, optimized_prompt])
    except Exception as e:  # pragma: no cover
        logger.warning(f"Failed to append inference logs: {e}")
    # MLflow logging
    if os.getenv("DISABLE_MLFLOW_INFERENCE") == '1':
        return
    try:
        # Ensure there is an active run (fallback if parent not set)
        if _parent_run_id:
            mlflow.start_run(run_name=f"infer_{channel}", nested=True)
        elif mlflow.active_run() is None:
            mlflow.start_run(run_name="inference_server_fallback", nested=False)
        mlflow.log_param("channel", channel)
        mlflow.log_param("artifact_path", artifact or "unknown")
        mlflow.log_param("initial_prompt_len", len(initial_prompt))
        mlflow.log_param("optimized_prompt_len", len(optimized_prompt))
        # Short snippets (truncate for param display)
        snippet_limit = int(os.getenv("PROMPT_SNIPPET_LIMIT", "200"))
        ip_snip = (initial_prompt[:snippet_limit] + ("…" if len(initial_prompt) > snippet_limit else "")).replace('\n', ' ')
        op_snip = (optimized_prompt[:snippet_limit] + ("…" if len(optimized_prompt) > snippet_limit else "")).replace('\n', ' ')
        mlflow.log_param("initial_prompt_snippet", ip_snip)
        mlflow.log_param("optimized_prompt_snippet", op_snip)
        # Hashes for integrity / dedup
        try:
            import hashlib
            mlflow.set_tags({
                "initial_prompt_sha256": hashlib.sha256(initial_prompt.encode('utf-8')).hexdigest(),
                "optimized_prompt_sha256": hashlib.sha256(optimized_prompt.encode('utf-8')).hexdigest(),
            })
        except Exception:  # pragma: no cover
            pass
        # Log prompt text artifacts
        mlflow.log_text(initial_prompt, artifact_file=f"prompts/{ts_safe}_initial.txt")
        mlflow.log_text(optimized_prompt, artifact_file=f"prompts/{ts_safe}_optimized.txt")
        pair_content = (
            "INITIAL_PROMPT:\n" + initial_prompt + "\n\n" +
            "OPTIMIZED_PROMPT:\n" + optimized_prompt + "\n"
        )
        mlflow.log_text(pair_content, artifact_file=f"pair/{ts_safe}_pair.txt")
        mlflow.log_metric("optimized_length", len(optimized_prompt))
        # Upload aggregated dataset file (could be large; optional control)
        if _inference_jsonl_path and os.getenv("DISABLE_DATASET_ARTIFACT_UPLOAD") != '1':
            try:
                mlflow.log_artifact(str(_inference_jsonl_path), artifact_path="inference_dataset")
            except Exception as e:  # pragma: no cover
                logger.warning(f"Failed to log JSONL dataset artifact: {e}")
        if _inference_pairs_csv_path and os.getenv("DISABLE_DATASET_ARTIFACT_UPLOAD") != '1':
            try:
                mlflow.log_artifact(str(_inference_pairs_csv_path), artifact_path="inference_dataset")
            except Exception as e:  # pragma: no cover
                logger.warning(f"Failed to log pairs CSV artifact: {e}")
    except Exception as e:  # pragma: no cover
        logger.warning(f"MLflow logging failed for inference: {e}")
    finally:
        # End nested run (not parent)
        try:
            active = mlflow.active_run()
            if active and active.info.run_id != _parent_run_id:
                mlflow.end_run()
        except Exception:  # pragma: no cover
            pass

# -------- Utility / Loading Logic --------

def _resolve_program_path() -> Optional[str]:
    explicit = os.getenv("PROGRAM_PATH")
    if explicit:
        return explicit
    # default best path
    candidate = Path(config.artifacts_dir) / "best_program.json"
    if candidate.exists():
        return str(candidate)
    # fallback to legacy default
    legacy = Path(config.save_path)
    return str(legacy) if legacy.exists() else None


def _load_or_reload_if_needed(force: bool = False):
    global _program, _program_path_used, _program_mtime
    with _program_lock:
        path = _resolve_program_path()
        if path is None:
            raise FileNotFoundError("No program artifact found. Run the optimization pipeline first.")
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            # Could be pointer file that disappeared; try fresh load which will raise
            mtime = time.time()
        if force or _program is None or path != _program_path_used or mtime > _program_mtime:
            logger.info(f"Loading program from artifact: {path}")
            _program = load_program(path=path, best=True)
            _program_path_used = path
            _program_mtime = mtime
            logger.info("Program loaded / reloaded successfully.")
        return _program


def _reload_watcher(interval: int):  # background thread
    logger.info(f"Started reload watcher every {interval}s")
    while not _stop_reload.wait(interval):
        try:
            _load_or_reload_if_needed(force=False)
        except Exception as e:  # pragma: no cover
            logger.warning(f"Reload watcher encountered error: {e}")
    logger.info("Reload watcher stopped.")

# -------- API Schemas --------

class OptimizeRequest(BaseModel):
    initial_prompt: str
    force_reload: bool | None = False

class OptimizeResponse(BaseModel):
    optimized_prompt: str
    artifact: Optional[str]

class HealthResponse(BaseModel):
    status: str
    artifact: Optional[str]

# -------- API Routes --------

@app.get("/health", response_model=HealthResponse)
async def health():
    try:
        _load_or_reload_if_needed()
        return HealthResponse(status="ok", artifact=_program_path_used)
    except Exception as e:  # pragma: no cover
        return HealthResponse(status=f"error: {e}", artifact=_program_path_used)

@app.post("/optimize", response_model=OptimizeResponse)
async def optimize(req: OptimizeRequest):
    try:
        _load_or_reload_if_needed(force=bool(req.force_reload))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Failed to load program: {e}")
    try:
        # run inference (program call is synchronous; offload not strictly needed unless latency high)
        with _program_lock:
            result = optimize_prompt(req.initial_prompt, program=_program)
        # Log inference pair
        _log_inference_pair(req.initial_prompt, result, channel="api")
        return OptimizeResponse(optimized_prompt=result, artifact=_program_path_used)
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

"""Telegram Bot Logic (integrated into FastAPI event loop)."""

async def _tg_start(update: Update, context: ContextTypes.DEFAULT_TYPE):  # type: ignore
    await update.message.reply_text("Send me a prompt and I'll optimize it.")

async def _tg_optimize(update: Update, context: ContextTypes.DEFAULT_TYPE):  # type: ignore
    if not update.message or not update.message.text:
        return
    prompt_text = update.message.text.strip()
    try:
        _load_or_reload_if_needed()
        with _program_lock:
            optimized = optimize_prompt(prompt_text, program=_program)
            _log_inference_pair(prompt_text, optimized, channel="telegram")
        await update.message.reply_text(optimized)  # Only optimized prompt
    except Exception as e:  # pragma: no cover
        await update.message.reply_text(f"Error: {e}")

async def _start_telegram_bot_in_loop(token: str):
    global _telegram_app
    if Application is None:
        logger.warning("python-telegram-bot not installed; skipping Telegram bot startup.")
        return
    if _telegram_app is not None:
        logger.info("Telegram bot already initialized.")
        return
    _telegram_app = Application.builder().token(token).build()
    _telegram_app.add_handler(CommandHandler("start", _tg_start))
    _telegram_app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), _tg_optimize))
    await _telegram_app.initialize()
    await _telegram_app.start()
    await _telegram_app.updater.start_polling()
    logger.info("Telegram bot started polling (event loop integration).")

async def _launch_telegram_with_retries(token: str):
    import asyncio
    max_retries = int(os.getenv("TELEGRAM_MAX_RETRIES", "3") or 3)
    base_delay = float(os.getenv("TELEGRAM_RETRY_BASE_DELAY", "5") or 5)
    for attempt in range(1, max_retries + 1):
        try:
            await _start_telegram_bot_in_loop(token)
            return
        except Exception as e:  # pragma: no cover
            if attempt == max_retries:
                logger.warning(f"Telegram bot failed to start after {attempt} attempts: {e}")
            else:
                delay = base_delay * attempt
                logger.warning(f"Telegram start attempt {attempt} failed: {e}. Retrying in {delay:.1f}s ...")
                await asyncio.sleep(delay)

async def _shutdown_telegram_bot():
    if _telegram_app is None:
        return
    try:
        await _telegram_app.updater.stop()
        await _telegram_app.stop()
        await _telegram_app.shutdown()
        logger.info("Telegram bot shutdown complete.")
    except Exception as e:  # pragma: no cover
        logger.warning(f"Error shutting down Telegram bot: {e}")

# -------- FastAPI Startup / Shutdown --------

@app.on_event("startup")
async def on_startup():
    # Initial load
    try:
        _load_or_reload_if_needed(force=True)
    except Exception as e:  # pragma: no cover
        logger.warning(f"Startup program load failed: {e}")
    # Initialize inference logging infra
    try:
        _init_inference_logging()
    except Exception as e:  # pragma: no cover
        logger.warning(f"Failed to init inference logging: {e}")

    # Start reload watcher if interval provided
    interval = int(os.getenv("RELOAD_INTERVAL_SECONDS", "0") or 0)
    if interval > 0:
        global _reload_thread
        _reload_thread = threading.Thread(target=_reload_watcher, args=(interval,), daemon=True, name="ProgramReloadWatcher")
        _reload_thread.start()

    # Start Telegram bot if token provided
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if token and os.getenv("TELEGRAM_DISABLE") != '1':
        try:
            import asyncio
            # Fire and forget; retries handled internally so startup not blocked.
            asyncio.create_task(_launch_telegram_with_retries(token))
        except Exception as e:  # pragma: no cover
            logger.warning(f"Failed to schedule Telegram bot startup: {e}")

@app.on_event("shutdown")
async def on_shutdown():
    _stop_reload.set()
    if _reload_thread and _reload_thread.is_alive():
        _reload_thread.join(timeout=5)
    await _shutdown_telegram_bot()

# -------- CLI Entrypoint (optional) --------
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host=host, port=port, reload=False)
