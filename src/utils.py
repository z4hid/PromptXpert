"""Dataset & config helpers for PromptXpert."""

import random
import pandas as pd
import dspy
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import logging
from .config import PromptXpertConfig, DataConfig

logger = logging.getLogger(__name__)


def set_reproducibility_seeds(seed: int = 42):
    """Set seeds for reproducibility across different libraries."""
    random.seed(seed)
    np.random.seed(seed)
    # Set DSPy seed if available
    if hasattr(dspy.settings, 'seed'):
        dspy.settings.seed = seed
    
    logger.info(f"Set reproducibility seed to {seed}")


def load_dataset(config: DataConfig) -> pd.DataFrame:
    """Load the dataset from CSV file.
    
    Args:
        config: Data configuration
        
    Returns:
        Loaded DataFrame
        
    Raises:
        FileNotFoundError: If dataset file doesn't exist
        ValueError: If required columns are missing
    """
    dataset_path = Path(config.dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    
    required_columns = ['initial_prompt', 'optimized_prompt']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Clean the data
    df = df.dropna(subset=required_columns)
    df['initial_prompt'] = df['initial_prompt'].astype(str).str.strip()
    df['optimized_prompt'] = df['optimized_prompt'].astype(str).str.strip()
    
    # Remove empty prompts
    df = df[(df['initial_prompt'] != '') & (df['optimized_prompt'] != '')]
    
    logger.info(f"Loaded dataset with {len(df)} examples from {dataset_path}")
    return df


def create_dspy_examples(df: pd.DataFrame) -> List[dspy.Example]:
    """Convert DataFrame to DSPy Examples.
    
    Args:
        df: DataFrame with prompt data
        
    Returns:
        List of DSPy Examples
    """
    examples = []
    for _, row in df.iterrows():
        example = dspy.Example(
            initial_prompt=row['initial_prompt'],
            optimized_prompt=row['optimized_prompt']
        ).with_inputs('initial_prompt')
        examples.append(example)
    
    logger.info(f"Created {len(examples)} DSPy examples")
    return examples


def split_dataset(
    examples: List[dspy.Example], 
    config: DataConfig
) -> Tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
    """Split dataset into train, validation, and test sets with shuffling.
    
    Args:
        examples: List of DSPy examples
        config: Data configuration with split ratios
        
    Returns:
        Tuple of (train_examples, val_examples, test_examples)
    """
    # Set seed for reproducible splitting
    random.seed(config.random_seed)
    
    # Shuffle the examples
    shuffled_examples = examples.copy()
    random.shuffle(shuffled_examples)
    
    total_size = len(shuffled_examples)
    train_size = int(config.train_ratio * total_size)
    val_size = int(config.val_ratio * total_size)
    
    # Split the data
    train_examples = shuffled_examples[:train_size]
    val_examples = shuffled_examples[train_size:train_size + val_size]
    test_examples = shuffled_examples[train_size + val_size:]
    
    logger.info(f"Dataset split - Train: {len(train_examples)}, "
                f"Val: {len(val_examples)}, Test: {len(test_examples)}")
    
    return train_examples, val_examples, test_examples


def augment_dataset(
    examples: List[dspy.Example], 
    augmentation_factor: float = 0.5
) -> List[dspy.Example]:
    """Augment dataset by creating degraded versions of good prompts.
    
    This creates synthetic training examples by taking optimized prompts
    and creating simplified versions as 'initial' prompts.
    
    Args:
        examples: Original examples
        augmentation_factor: Fraction of examples to augment (0.0 to 1.0)
        
    Returns:
        Augmented list of examples
    """
    if augmentation_factor <= 0:
        return examples
    
    augmented_examples = examples.copy()
    num_to_augment = int(len(examples) * augmentation_factor)
    
    # Select random examples to augment
    examples_to_augment = random.sample(examples, min(num_to_augment, len(examples)))
    
    for example in examples_to_augment:
        # Create a simplified version of the optimized prompt
        optimized = example.optimized_prompt
        
        # Simple degradation strategies
        degraded_prompts = [
            # Remove details
            " ".join(optimized.split()[:len(optimized.split())//2]),
            # Make more generic
            "Help me with " + optimized.split()[0].lower() if optimized.split() else optimized,
            # Truncate
            optimized[:len(optimized)//3] if len(optimized) > 30 else optimized
        ]
        
        for degraded in degraded_prompts:
            if degraded.strip() and degraded != optimized:
                augmented_example = dspy.Example(
                    initial_prompt=degraded.strip(),
                    optimized_prompt=optimized
                ).with_inputs('initial_prompt')
                augmented_examples.append(augmented_example)
    
    logger.info(f"Augmented dataset from {len(examples)} to {len(augmented_examples)} examples")
    return augmented_examples


def prepare_dataset(config: PromptXpertConfig) -> Tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
    """Complete dataset preparation pipeline.
    
    Args:
        config: Full configuration
        
    Returns:
        Tuple of (train_examples, val_examples, test_examples)
    """
    # Set reproducibility
    set_reproducibility_seeds(config.data.random_seed)
    
    # Load and convert dataset
    df = load_dataset(config.data)
    examples = create_dspy_examples(df)
    
    # Optionally augment dataset
    if len(examples) < 20:  # Only augment if dataset is small
        examples = augment_dataset(examples, augmentation_factor=0.3)
    
    # Split dataset
    train_examples, val_examples, test_examples = split_dataset(examples, config.data)
    
    return train_examples, val_examples, test_examples


def save_dataset_info(
    train_examples: List[dspy.Example],
    val_examples: List[dspy.Example], 
    test_examples: List[dspy.Example],
    output_path: str
):
    """Save dataset information for tracking and reproducibility.
    
    Args:
        train_examples: Training examples
        val_examples: Validation examples
        test_examples: Test examples
        output_path: Path to save the info
    """
    info = {
        'train_size': len(train_examples),
        'val_size': len(val_examples),
        'test_size': len(test_examples),
        'total_size': len(train_examples) + len(val_examples) + len(test_examples),
        'sample_train_examples': [
            {'initial': ex.initial_prompt, 'optimized': ex.optimized_prompt}
            for ex in train_examples[:3]
        ]
    }
    
    import json
    with open(output_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"Saved dataset info to {output_path}")


def validate_dataset_quality(examples: List[dspy.Example]) -> Dict[str, Any]:
    """Validate dataset quality and return statistics.
    
    Args:
        examples: List of examples to validate
        
    Returns:
        Dictionary with quality metrics
    """
    if not examples:
        return {'error': 'Empty dataset'}
    
    stats = {
        'total_examples': len(examples),
        'avg_initial_length': np.mean([len(ex.initial_prompt) for ex in examples]),
        'avg_optimized_length': np.mean([len(ex.optimized_prompt) for ex in examples]),
        'length_ratio': 0,
        'unique_initials': len(set(ex.initial_prompt for ex in examples)),
        'duplicate_pairs': 0
    }
    
    # Calculate length ratio
    if stats['avg_initial_length'] > 0:
        stats['length_ratio'] = stats['avg_optimized_length'] / stats['avg_initial_length']
    
    # Check for duplicate pairs
    pairs = set()
    duplicates = 0
    for ex in examples:
        pair = (ex.initial_prompt, ex.optimized_prompt)
        if pair in pairs:
            duplicates += 1
        pairs.add(pair)
    stats['duplicate_pairs'] = duplicates
    
    # Quality warnings
    warnings = []
    if stats['total_examples'] < 10:
        warnings.append("Very small dataset (< 10 examples)")
    if stats['length_ratio'] < 1.2:
        warnings.append("Optimized prompts not significantly longer than originals")
    if stats['duplicate_pairs'] > 0:
        warnings.append(f"Found {stats['duplicate_pairs']} duplicate pairs")
    if stats['unique_initials'] / stats['total_examples'] < 0.8:
        warnings.append("Low diversity in initial prompts")
    
    stats['warnings'] = warnings
    return stats
