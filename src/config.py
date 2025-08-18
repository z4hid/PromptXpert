from dataclasses import dataclass
import os

@dataclass
class Config:
    main_model: str = "gemini/gemini-2.5-flash-lite"
    judge_model: str = "gemini/gemini-2.5-flash-lite"
    main_temperature: float = 0.7
    judge_temperature: float = 0.0
    max_bootstrapped_demos: int = 2
    max_labeled_demos: int = 2
    auto_level: str = "light"
    minibatch_size: int = 2
    num_threads: int = 1
    seed: int = 42
    dataset_csv: str = "data/dataset.csv"
    artifacts_dir: str = "artifacts"
    save_state_only: bool = True  # state-only JSON (safer, readable)
    save_whole_program: bool = True  # also persist whole program directory (dspy>=2.6)
    # Legacy single-file paths (still produced for backward compatibility)
    save_path: str = "./prompt_xpert_program_optimized.json"
    metadata_path: str = "./optimization_metadata.json"

    def ensure_artifacts_dir(self):
        os.makedirs(self.artifacts_dir, exist_ok=True)
        return self.artifacts_dir

    def set_dataset(self, path: str):
        """Update dataset path (CLI override)."""
        self.dataset_csv = path
        return self.dataset_csv

config = Config()
