import os
import csv
import pandas as pd
import dspy
from typing import List
from .logging_setup import logger

def ensure_dataset(csv_file_path: str, force_recreate: bool = False):
    rows = [
        ("Write a concise summary of the provided text.",
         "Summarize the text in under 50 words, highlighting key findings and excluding jargon."),
        ("Generate a Python function to sort a list.",
         "Create a Python function named `sort_list` that takes a list of integers as input and returns a new list sorted in ascending order."),
        ("Explain recursion to a 5-year-old.",
         "Using the analogy of nested Russian dolls, explain recursion simply for a child."),
        ("Translate 'hello' to French.",
         "Provide the formal translation of the English word 'hello' into French."),
        ("Write a thank you email to a job interviewer.",
         "Compose a formal, concise thank you email to a job interviewer within 100 words, reiterating interest in the position."),
        ("Describe the capital of France.",
         "Provide a brief, factual description of Paris focusing on its governmental role and key cultural aspects."),
        ("Summarize the plot of Hamlet.",
         "Concisely summarize the main plot points of Shakespeare's Hamlet, including key characters and the central conflict, in under 75 words."),
    ]
    # Ensure parent dir exists
    os.makedirs(os.path.dirname(csv_file_path) or '.', exist_ok=True)
    if force_recreate:
        logger.info(f"Force recreating dataset at {csv_file_path}")
    if force_recreate or not os.path.exists(csv_file_path):
        logger.info(f"Creating default dataset at {csv_file_path}")
        with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["initial_prompt", "optimized_prompt"])
            for r in rows:
                writer.writerow(r)
    else:
        logger.info(f"Using existing dataset: {csv_file_path}")

def load_examples(csv_file_path: str) -> List[dspy.Example]:
    df = pd.read_csv(csv_file_path)
    df = df.dropna(subset=["initial_prompt", "optimized_prompt"])
    examples: List[dspy.Example] = []
    for _, row in df.iterrows():
        examples.append(
            dspy.Example(
                initial_prompt=str(row["initial_prompt"]).strip(),
                optimized_prompt=str(row["optimized_prompt"]).strip(),
            ).with_inputs("initial_prompt")
        )
    return examples

def split_train_dev(examples: List[dspy.Example], train_ratio: float = 0.8):
    n = len(examples)
    k = max(1, int(train_ratio * n))
    trainset = examples[:k]
    devset = examples[k:] if k < n else examples[:1]
    return trainset, devset
