<div align="center">

# PromptXpert

Automated prompt optimization pipeline & service built on **DSPy** (MIPROv2 teleprompting) with multi‑criteria self‑judging, artifact versioning, MLflow / DagsHub tracking, a REST API, and optional Telegram bot interface.

</div>

---

## Table of Contents
1. Overview
2. Features
3. Architecture & Directory Layout
4. Getting Started
5. CLI Usage
6. API (FastAPI) Usage
7. Telegram Bot
8. Artifacts & Versioning
9. Tracking (MLflow / DagsHub)
10. Configuration & Environment Variables
11. Dataset
12. Internals (Program, Metric & Selection Logic)
13. Inference Logging Outputs
14. Extending & Customization
15. Roadmap
16. Contributing
17. Troubleshooting
18. License

---

## 1. Overview
PromptXpert learns to rewrite user prompts into clearer, more specific, and more effective versions. It compiles an optimization program using DSPy **MIPROv2**, evaluates quality via a multi‑criteria judging signature, and versions every compiled program so you can reuse or deploy the best performing artifact.

## 2. Features
- Teleprompting with `MIPROv2` (DSPy) for iterative prompt improvement.
- Multi‑criteria judging (clarity, specificity, completeness, effectiveness) -> weighted average score.
- Dual artifact formats: state‑only JSON and whole-program directory (architecture + state).
- Automatic best program tracking (`artifacts/best_program.json`).
- Robust dataset bootstrapping & malformed CSV recovery.
- MLflow / DagsHub experiment tracking (nested runs for compilation & evaluation).
- FastAPI service with hot‑reload of improved artifacts.
- Optional Telegram bot responding with optimized prompts only.
- Structured inference logging to CSV + JSONL + pairs CSV.
- Rate‑limited LM wrapper with retries (tenacity + litellm).

## 3. Architecture & Directory Layout
```
promptxpert.py              # CLI entry (train / infer)
app.py                      # FastAPI + optional Telegram bot server
src/
  config.py                 # Config dataclass & flags
  data_utils.py             # Dataset ensure/load/split helpers
  lm_init.py                # LM initialization + rate limiting
  metrics.py                # Multi-criteria judge metric
  program.py                # DSPy signatures & optimization module
  pipeline.py               # End-to-end training + artifact versioning
  inference.py              # Loading & prompt optimization helpers
  logging_setup.py          # Logging + MLflow baseline setup
  tracking.py               # DagsHub / MLflow tracking helpers
artifacts/ (after training)
  promptxpert_<ts>_scoreXXXX.json / _meta.json
  promptxpert_<ts>_scoreXXXX_dir/ (whole program)
  best_program.json / best_program_meta.json
  inference_log.csv / inference_dataset.jsonl / inference_pairs.csv
data/
  dataset.csv               # Default seed dataset (auto-created)
```

## 4. Getting Started
Prerequisites:
- Python 3.10+
- A Gemini (or compatible) API key in `GEMINI_API_KEY` (defaults use Gemini flash-lite models)

Install dependencies:
```bash
pip install -r requirements.txt
```
Export API key (example):
```bash
export GEMINI_API_KEY=YOUR_KEY
```
Optional (local MLflow UI):
```bash
mlflow ui --port 5000
```

## 5. CLI Usage
Train (compile + evaluate + version artifacts):
```bash
python promptxpert.py --mode train
```
Custom dataset & immediate sample optimization:
```bash
python promptxpert.py --mode train --dataset data/dataset.csv --prompt "Improve: outline a marketing launch plan for an eco-friendly gadget"
```
Inference using best artifact:
```bash
python promptxpert.py --mode infer --prompt "Improve: write code to reverse a list"
```
Specify an explicit artifact (file or directory):
```bash
python promptxpert.py --mode infer --model_path artifacts/promptxpert_20250818-195258_score93-7500.json --prompt "Improve: summarize quarterly sales report"
```
Disable auto best selection:
```bash
python promptxpert.py --mode infer --no_best --prompt "Improve: craft a polite follow-up email"
```

## 6. API (FastAPI) Usage
Start server:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
Health check:
```bash
curl http://localhost:8000/health
```
Optimize (POST /optimize):
```bash
curl -X POST http://localhost:8000/optimize \
  -H 'Content-Type: application/json' \
  -d '{"initial_prompt":"Write a thank you email"}'
```
Response:
```json
{ "optimized_prompt": "...", "artifact": "artifacts/best_program.json" }
```
Hot Reload: set `RELOAD_INTERVAL_SECONDS>0` to periodically reload a newer `best_program.json`.

## 7. Telegram Bot
Set `TELEGRAM_BOT_TOKEN` and run the same FastAPI server; the bot will start automatically (unless `TELEGRAM_DISABLE=1`). Each inbound message is treated as `initial_prompt` and replied with only the optimized prompt.

## 8. Artifacts & Versioning
Each training run creates a timestamped base name: `promptxpert_<YYYYMMDD-HHMMSS>_score<metric>`.

Generated items (depending on config flags):
- State JSON: `<basename>.json` (fast, human-readable)
- Whole program directory: `<basename>_dir/` (loadable with `dspy.load`)
- Metadata JSON: `<basename>_meta.json`
- Best copies: `best_program.json`, `best_program_meta.json`
- Legacy: `prompt_xpert_program_optimized.json`, `optimization_metadata.json`

Selection Logic (in inference):
1. If `--model_path` provided: load that (file, dir, or pointer JSON containing `redirect_to_whole_program_dir`).
2. Else if `--no_best` not set and `artifacts/best_program.json` exists: load it.
3. Else fallback to legacy single-file state.

## 9. Tracking (MLflow / DagsHub)
Set these to enable remote tracking on DagsHub:
```bash
export DAGSHUB_REPO_OWNER=<owner>
export DAGSHUB_REPO_NAME=PromptXpert
export DAGSHUB_TOKEN=<token>
export MLFLOW_EXPERIMENT_NAME=PromptXpert-Optimization   # optional
```
Behavior:
- Configures `MLFLOW_TRACKING_URI` = `https://dagshub.com/<owner>/<repo>.mlflow`.
- Creates parent `pipeline` run with nested runs: `MIPROv2 Compilation`, `Optimized Program Evaluation`.
- Logs artifacts under `program/` and `best/` plus metrics & params.
Omit the token to use local `./mlruns`.

## 10. Configuration & Environment Variables
Core (see `src/config.py`):
- `main_model`, `judge_model`, temps, `auto_level` (light|medium|heavy), demo counts, minibatch size.
- `save_state_only`, `save_whole_program` (both True by default).

Environment variables:
- `GEMINI_API_KEY` (required)
- `PROGRAM_PATH` (override artifact path for API)
- `RELOAD_INTERVAL_SECONDS` (API background reload interval)
- `TELEGRAM_BOT_TOKEN` (enable bot) / `TELEGRAM_DISABLE=1` (disable)
- `DISABLE_MLFLOW_INFERENCE=1` (skip logging for API/bot calls)
- `DISABLE_DATASET_ARTIFACT_UPLOAD=1` (avoid re-uploading growing JSONL/CSV)
- `PROMPT_SNIPPET_LIMIT` (truncate snippet length in MLflow params)

## 11. Dataset
Default path: `data/dataset.csv` (created automatically if missing). Columns:
- `initial_prompt`
- `optimized_prompt`

Malformed CSV resilience: extra columns are merged for `optimized_prompt`; bad lines skipped.
Override with CLI flag: `--dataset path/to/custom.csv`.

## 12. Internals (Program, Metric & Selection Logic)
- Program (`PromptXpertProgram`): wraps a `ChainOfThought` over `PromptOptimization` signature.
- Metric: `MultiCriteriaPromptMetric` executes a judging signature returning four sub-scores -> weighted average.
- Saving modes: state-only (JSON) vs whole-program directory (DSPy >= 2.6) – both enabled by default.

## 13. Inference Logging Outputs
API / Telegram calls append to:
- `artifacts/inference_log.csv` (wide log with channel + timestamps)
- `artifacts/inference_dataset.jsonl` (one JSON per inference)
- `artifacts/inference_pairs.csv` (initial, optimized only)
Logged (if enabled) to MLflow as nested runs with prompt length metrics & SHA256 hashes.

## 14. Extending & Customization
- Add evaluation dimensions: edit `metrics.py` (update weights & signature fields).
- Swap provider: adjust `config.py` models + environment key variable(s).
- Larger optimization: increase dataset, adjust `auto_level` and demo limits.
- Integrate retrieval / context: wrap program forward pass with retrieval step before optimization.

## 15. Roadmap
- Pluggable metric registry / config file weights.
- Retrieval augmented optimization.
- Multi‑model or ensemble judging.
- Cost tracking & budget abort thresholds.
- Lightweight web UI for interactive optimization.

## 16. Contributing
1. Fork & branch (`feat/<name>`).
2. Run a training cycle (`python promptxpert.py --mode train`).
3. Ensure artifacts & logging behave as expected.
4. Add / adapt tests (if introduced) & update docs.
5. Open a PR with context + before/after notes.

## 17. Troubleshooting
| Issue | Hint |
|-------|------|
| Missing API key | Export `GEMINI_API_KEY` before running. |
| Rate limits / retries | Reduce minibatch size or raise wait interval; tenacity handles transient failures. |
| Best program not updating | Check write perms on `artifacts/` & console logs. |
| Pointer load fails | Ensure the referenced directory in `redirect_to_whole_program_dir` exists. |
| MLflow errors | Remote tracking vars optional; runs fallback locally if not set. |

## 18. License
See `LICENSE` (add one if missing). If absent, the repository is effectively proprietary—consider adding MIT/Apache-2.0.

---
### Quick Start (Shortest Path)
```bash
export GEMINI_API_KEY=YOUR_KEY
pip install -r requirements.txt
python promptxpert.py --mode train --prompt "Improve: generate a concise FAQ for a travel booking site"
python promptxpert.py --mode infer --prompt "Improve: draft a data retention policy overview"
```

---
Happy optimizing! 🚀
