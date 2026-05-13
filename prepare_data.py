"""
Task-01: Dataset Preparation Utility
=====================================
Helpers to clean, split, and preview your training corpus.
Run this before train.py to prepare data/train.txt (and optionally data/val.txt).
"""

import os
import re
import argparse
import random
import logging
from pathlib import Path

logging.basicConfig(format="%(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Text cleaning
# ──────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Basic cleaning: normalise whitespace, remove non-printable chars."""
    text = re.sub(r"[^\x00-\x7F]+", " ", text)          # strip non-ASCII
    text = re.sub(r"[ \t]+", " ", text)                  # collapse spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)               # max 2 blank lines
    return text.strip()


# ──────────────────────────────────────────────
# Train / val split
# ──────────────────────────────────────────────
def split_and_save(text: str, output_dir: str, val_ratio: float = 0.1):
    lines = text.split("\n")
    random.shuffle(lines)

    split_idx = int(len(lines) * (1 - val_ratio))
    train_lines = lines[:split_idx]
    val_lines   = lines[split_idx:]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_path = out / "train.txt"
    val_path   = out / "val.txt"

    train_path.write_text("\n".join(train_lines), encoding="utf-8")
    val_path.write_text("\n".join(val_lines), encoding="utf-8")

    logger.info(f"Train examples : {len(train_lines):,}  →  {train_path}")
    logger.info(f"Val   examples : {len(val_lines):,}  →  {val_path}")


# ──────────────────────────────────────────────
# Built-in sample corpus (demo)
# ──────────────────────────────────────────────
SAMPLE_CORPUS = """
The quick brown fox jumps over the lazy dog.
She sells seashells by the seashore and the shells she sells are seashells.
To be or not to be, that is the question whether tis nobler in the mind to suffer.
In the beginning was the word and the word was with light and the light was knowledge.
Artificial intelligence is transforming the way we live, work, and communicate.
The transformer architecture revolutionised natural language processing in 2017.
Fine-tuning a pre-trained language model adapts its knowledge to a specific domain.
Generative models learn the underlying distribution of text and sample from it.
Deep learning relies on gradient-based optimisation and large-scale datasets.
The attention mechanism allows models to focus on relevant parts of the input.
Language models predict the probability of the next token given all previous tokens.
Perplexity measures how well a probability model predicts a sample of held-out text.
""" * 50   # repeat to create a larger demo corpus


def prepare_sample(output_dir: str):
    logger.info("Generating sample corpus …")
    text = clean_text(SAMPLE_CORPUS)
    split_and_save(text, output_dir)
    logger.info("Sample data ready. Edit data/train.txt with your own content.")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Prepare dataset for GPT-2 fine-tuning")
    p.add_argument("--input_file",  default=None,   help="Path to raw .txt file (omit to use built-in sample)")
    p.add_argument("--output_dir",  default="data", help="Directory for train.txt / val.txt")
    p.add_argument("--val_ratio",   type=float, default=0.1, help="Fraction of data for validation (0–1)")
    p.add_argument("--no_clean",    action="store_true",     help="Skip text cleaning step")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.input_file is None:
        prepare_sample(args.output_dir)
    else:
        assert os.path.isfile(args.input_file), f"Not found: {args.input_file}"
        text = Path(args.input_file).read_text(encoding="utf-8")
        if not args.no_clean:
            text = clean_text(text)
            logger.info("Text cleaned.")
        split_and_save(text, args.output_dir, args.val_ratio)
