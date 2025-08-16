import sys
from pathlib import Path


def test_structure_exists():
    root = Path(__file__).resolve().parents[1]
    assert (root / 'src' / '__init__.py').exists(), 'src/__init__.py missing'
    assert (root / 'data' / 'prompts_dataset.csv').exists(), 'prompts_dataset.csv missing'

