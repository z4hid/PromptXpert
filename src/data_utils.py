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
    """Load examples from a CSV with robust fallbacks.

    Handles cases where rows contain extra commas or malformed quoting by:
      1. First attempting a standard pandas parse.
      2. On failure, re-reading with the python engine + on_bad_lines='skip'.
      3. If resulting frame has >2 columns, merge trailing columns into
         a single `optimized_prompt` column (joined by commas).
    """
    try:
        df = pd.read_csv(csv_file_path)
    except pd.errors.ParserError as e:
        logger.warning(f"Primary CSV parse failed ({e}); attempting tolerant parse.")
        df = pd.read_csv(
            csv_file_path,
            engine="python",
            on_bad_lines="skip",
        )
    # If required columns missing but we have at least 2 columns, attempt salvage
    required = {"initial_prompt", "optimized_prompt"}
    if not required.issubset(df.columns):
        if len(df.columns) >= 2:
            logger.warning("CSV columns malformed; reconstructing expected schema.")
            first_col = df.columns[0]
            # Merge remaining columns into optimized prompt text
            df["initial_prompt"] = df[first_col].astype(str)
            df["optimized_prompt"] = df[df.columns[1:]].astype(str).apply(lambda r: ",".join([c for c in r if c and c != 'nan']), axis=1)
        else:
            raise ValueError(f"Dataset at {csv_file_path} does not contain enough columns to build examples.")
    df = df.dropna(subset=["initial_prompt", "optimized_prompt"])
    examples: List[dspy.Example] = []
    for _, row in df.iterrows():
        init_txt = str(row["initial_prompt"]).strip()
        opt_txt = str(row["optimized_prompt"]).strip()
        if not init_txt or not opt_txt:
            continue
        examples.append(
            dspy.Example(
                initial_prompt=init_txt,
                optimized_prompt=opt_txt,
            ).with_inputs("initial_prompt")
        )
    if not examples:
        raise ValueError(f"No valid examples parsed from {csv_file_path} after cleaning.")
    return examples

def split_train_dev(examples: List[dspy.Example], train_ratio: float = 0.8):
    n = len(examples)
    k = max(1, int(train_ratio * n))
    trainset = examples[:k]
    devset = examples[k:] if k < n else examples[:1]
    return trainset, devset
