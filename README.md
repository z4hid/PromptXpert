# PromptXpert DSPy Optimization

Modular DSPy-based system for optimizing user prompts using automated teleprompting (MIPROv2) and multi-criteria self-judging. The project compiles improved prompting strategies, evaluates them, and versions each compiled program as an artifact so you can reuse or pick the best performing version for inference.

## Key Features
- **DSPy Teleprompting**: Uses `MIPROv2` to iteratively improve a prompt-optimization module.
- **Multi-Criteria Judging**: Clarity, specificity, completeness, effectiveness (weighted average) via a judging LM.
- **Artifact Versioning**: Each training run creates timestamped artifacts (state JSON and/or whole-program dir) + metadata.
- **Best Model Tracking**: The highest scoring program is copied to `artifacts/best_program.json` + meta file.
- **Modular Layout**: Clean separation of config, data utils, metrics, LM init, pipeline, inference.
- **CLI Modes**: Train (optimize & evaluate) or infer (reuse best or a specific saved program).
- **MLflow Integration**: Logs parameters, metrics, traces (if supported) with optional autologging.

## Directory Structure
```
promptxpert.py              # Entry CLI (train / infer)
src/
  config.py                 # Configuration dataclass
  logging_setup.py          # Logging + MLflow autolog configuration
  lm_init.py                # Rate-limited LM wrapper + initialization
  data_utils.py             # Dataset creation & loading helpers
  program.py                # DSPy signature + optimization module
  metrics.py                # Judge signature + multi-criteria metric
  pipeline.py               # End-to-end compile/eval/save + artifact versioning
  inference.py              # Loading and prompt optimization helpers
artifacts/                  # (Created after first successful train)
  promptxpert_<timestamp>_score<score>.json
  promptxpert_<timestamp>_score<score>_meta.json
  best_program.json         # Copy of highest scoring program
  best_program_meta.json    # Metadata for best program
```

## Requirements
Install dependencies (example):
```bash
pip install -r requirements.txt
```
Ensure environment variable for your model provider is set (example for Gemini):
```bash
export GEMINI_API_KEY=YOUR_KEY_HERE
```
Optionally run an MLflow server:
```bash
mlflow ui --port 5000
```
(Adjust tracking URI in code if needed.)

## Dataset
A small default CSV (`data/prompts_dataset.csv`) is auto-created if missing. You can override with `--dataset path/to/custom.csv`. Columns:
- `initial_prompt`
- `optimized_prompt`

## Training (Compile + Evaluate + Version Artifacts)
Run (uses dataset at `data/prompts_dataset.csv` by default):
```bash
python promptxpert.py --mode train
```
Optional: provide a sample prompt immediately optimized after training and custom dataset:
```bash
python promptxpert.py --mode train --dataset data/prompts_dataset.csv --prompt "Improve: outline a marketing launch plan for an eco-friendly gadget"
```
Artifacts produced in `artifacts/` may include:
- State-only JSON (readable) if `save_state_only=True`.
- Whole-program directory (architecture + state) if `save_whole_program=True`.
- Matching metadata JSON with score, sizes, config, timestamp.
- `best_program.json` (either a direct state file or a pointer JSON referencing a whole-program dir) + meta.
- Legacy root `prompt_xpert_program_optimized.json` + `optimization_metadata.json` for backward compatibility.

## Inference
Use the best stored program automatically:
```bash
python promptxpert.py --mode infer --prompt "Improve: write code to reverse a list"
```
Use a specific saved artifact:
```bash
python promptxpert.py --mode infer --model_path artifacts/promptxpert_20250101-120000_score0-9125.json --prompt "Improve: summarize quarterly sales report"
```
Force using legacy (ignore best selection logic):
```bash
python promptxpert.py --mode infer --no_best --prompt "Improve: craft a polite follow-up email"
```

## Program Selection Logic
1. If `--model_path` specified:
  - If it is a directory, load whole program via `dspy.load()`.
  - If it is a JSON pointer containing `redirect_to_whole_program_dir`, follow pointer.
  - Else treat as state JSON.
2. Else if `--no_best` not provided and `artifacts/best_program.json` exists:
  - Load as state or pointer (same logic as above).
3. Else fallback to legacy `prompt_xpert_program_optimized.json` (state-only).

## Metadata Fields
Each meta JSON contains:
- `avg_score` (float)
- `trainset_size`, `devset_size`
- `config` (serialized config dataclass)
- `program_file` (basename of associated program JSON)
- `timestamp`

## Saving Best Practices (DSPy)
This project implements both saving modes described in the DSPy tutorial:

- **State-only** (`save_program=False`): fast, human-readable JSON; requires recreating the program class before loading.
- **Whole Program** (`save_program=True`): serializes architecture + state into a directory; can be reloaded directly via `dspy.load(path/)` without reconstructing the module manually.

Configuration flags in `config.py` (toggle artifact types):
```python
save_state_only = True
save_whole_program = True
```
Disable either if you only want one type of artifact. When both are enabled we maintain parallel artifacts and always prefer the state JSON for `best_program.json` (with a pointer fallback if only whole-program exists).

## Testing
Add quick regression tests (suggested pattern):
```bash
pytest -k test_pipeline
```
Potential future tests: artifact creation, pointer loading, metric monotonicity.

## Contributing
1. Fork & branch (`feat/...`).
2. Run a training cycle to ensure artifacts generate.
3. Add/adjust tests.
4. Open PR with description + sample console output snippet.

## Roadmap
- Pluggable metrics via entry points.
- Optional retrieval augmentation.
- Multi-model ensemble scoring.
- Cost tracking & budget abort threshold.

## Extending
- Add more evaluation metrics (edit `metrics.py`).
- Swap model provider (adjust `config.py` + environment variables).
- Increase dataset and tune `auto_level` (light/medium/heavy) for more thorough optimization.

## Troubleshooting
- Missing API Key: Ensure `GEMINI_API_KEY` (or provider equivalent) is exported before running.
- Rate Limits: The `RateLimitedLM` uses tenacity retries; consider lowering batch sizes if limits persist.
- MLflow Not Found: Warnings are logged; functionality continues without tracking.
- Best Model Not Updating: Check write permissions to `artifacts/` and inspect console warnings.

## Safety & Costs
Prompt optimization can trigger multiple LM calls; monitor usage and costs in your provider dashboard.

## License
See `LICENSE` (inherits original project license if applicable).

## Quick Start
```bash
export GEMINI_API_KEY=YOUR_KEY
pip install -r requirements.txt
python promptxpert.py --mode train --prompt "Improve: generate a concise FAQ section for a travel booking site"
python promptxpert.py --mode infer --prompt "Improve: draft a data retention policy overview"
```

---
Happy optimizing!
