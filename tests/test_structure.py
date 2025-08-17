"""Test the overall structure and configuration without heavy dependencies."""

import sys
import json
from pathlib import Path


def test_file_structure():
    """Test that all expected files exist."""
    root = Path(__file__).resolve().parents[1]
    
    # Required files
    required_files = [
        'src/__init__.py',
        'src/config.py',
        'src/signature.py',
        'src/module.py',
        'src/utils.py',
        'src/metrics.py',
        'src/train.py',
        'src/eval.py',
        'src/cli.py',
        'src/mlflow_integration.py',
        'data/prompts_dataset.csv',
        'requirements.txt',
        '.env.template'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not (root / file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"Missing files: {missing_files}")
        return False
    
    print("✓ All required files exist")
    return True


def test_dataset_format():
    """Test that dataset has correct format."""
    root = Path(__file__).resolve().parents[1]
    dataset_path = root / 'data' / 'prompts_dataset.csv'
    
    with open(dataset_path, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        print("Dataset too small")
        return False
    
    header = lines[0].strip()
    expected_columns = ['initial_prompt', 'optimized_prompt']
    
    if not all(col in header for col in expected_columns):
        print(f"Missing required columns. Found: {header}")
        return False
    
    print(f"✓ Dataset format correct, {len(lines)-1} examples")
    return True


def test_config_structure():
    """Test configuration structure without importing heavy dependencies."""
    # Read the config file as text to check structure
    root = Path(__file__).resolve().parents[1]
    config_path = root / 'src' / 'config.py'
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    required_classes = [
        'ModelConfig',
        'OptimizationConfig', 
        'DataConfig',
        'MLflowConfig',
        'MetricsConfig',
        'PromptXpertConfig'
    ]
    
    missing_classes = []
    for class_name in required_classes:
        if f'class {class_name}' not in content:
            missing_classes.append(class_name)
    
    if missing_classes:
        print(f"Missing config classes: {missing_classes}")
        return False
    
    print("✓ Configuration structure correct")
    return True


def test_requirements():
    """Test that requirements file has expected dependencies."""
    root = Path(__file__).resolve().parents[1]
    req_path = root / 'requirements.txt'
    
    with open(req_path, 'r') as f:
        content = f.read()
    
    required_deps = ['dspy-ai', 'mlflow', 'pandas', 'numpy', 'groq']
    
    missing_deps = []
    for dep in required_deps:
        if dep not in content:
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"Missing dependencies: {missing_deps}")
        return False
    
    print("✓ Requirements file complete")
    return True


def main():
    """Run all structure tests."""
    print("=" * 50)
    print("PROMPTXPERT STRUCTURE VALIDATION")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_dataset_format,
        test_config_structure,
        test_requirements
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"✗ {test.__name__} failed")
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
    
    print("=" * 50)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ Structure validation successful!")
        return True
    else:
        print("✗ Structure validation failed!")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)